import type {
  FullConfig, FullResult, Reporter, Suite, TestCase, TestResult,
} from '@playwright/test/reporter';
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

/**
 * TG18.5 slice 4 — what a rendered browser run recorded, written so a release gate can read it.
 *
 * TG17.10's `browser_no_glue` gate reads `NOT_RUN` because "a deterministic backend rehearsal
 * cannot observe a rendered browser and must not award itself a gate on someone else's
 * evidence". This reporter is that someone else. It writes what the run actually did; it decides
 * nothing. `src/core/browser_evidence.py` decides, and refuses.
 *
 * **Three things are recorded, and the third is the one that makes the other two mean
 * something.**
 *
 *   1. Every test that ran, with its spec file and outcome. Counts, not prose.
 *   2. The artefacts the run produced, by name and digest. The images are gitignored, so the
 *      digest attests that a file of those bytes existed when the run finished - never that a
 *      human looked at it, and never that what it shows is correct.
 *   3. The **source digest of every spec in the suite**. A recording is a measurement of a
 *      particular set of tests. Change what a spec asserts and this recording is no longer
 *      about the suite that now exists, so the gate must return to `NOT_RUN` rather than let a
 *      pass earned by weaker assertions stand.
 *
 * **A partial run must not be able to qualify anything.** Playwright is routinely invoked on one
 * spec at a time during development, and such a run would otherwise write a recording that looks
 * like a green suite. The recording therefore names the specs it ran *and* the complete spec
 * inventory it found on disk, and the Python side requires them to agree.
 *
 * Nothing here is evidence about the world. It records apparatus behaviour in a browser.
 */

const SCHEMA = 'browser-run/v1';

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

/** Newline-normalised, so a Windows checkout and a POSIX one agree about the same source. */
function sourceDigest(path: string): string {
  const body = readFileSync(path);
  return sha256(Buffer.from(body.toString('utf-8').replace(/\r\n/g, '\n'), 'utf-8'));
}

function listFiles(directory: string): string[] {
  try {
    return readdirSync(directory)
      .filter(name => statSync(join(directory, name)).isFile())
      .sort();
  } catch {
    return [];
  }
}

interface SpecOutcome {
  spec: string;
  passed: number;
  failed: number;
  skipped: number;
}

export default class QualificationReporter implements Reporter {
  private readonly outcomes = new Map<string, SpecOutcome>();
  private started = '';
  private e2eDir = '';
  private repoRoot = '';

  onBegin(_config: FullConfig, _suite: Suite): void {
    this.started = new Date().toISOString();
    // From the working directory, the convention this suite's artefact paths already rely on.
    // `config.rootDir` is the common root Playwright computed for the run, which moves when the
    // run is filtered to one spec - exactly the case this record must not be fooled by.
    this.e2eDir = resolve(process.cwd(), 'e2e');
    this.repoRoot = resolve(process.cwd(), '..');
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    // `test.location.file` is the spec that declared it, which is what the record is about -
    // not the file a helper happens to live in.
    const spec = test.location.file.split(/[\\/]/).pop() ?? 'unknown';
    const row = this.outcomes.get(spec)
      ?? { spec, passed: 0, failed: 0, skipped: 0 };
    if (result.status === 'passed') row.passed += 1;
    else if (result.status === 'skipped') row.skipped += 1;
    else row.failed += 1;
    this.outcomes.set(spec, row);
  }

  onEnd(result: FullResult): void {
    const specs = listFiles(this.e2eDir).filter(name => name.endsWith('.spec.ts')).sort();
    const artefactDir = join(this.e2eDir, 'artifacts');

    const ran = [...this.outcomes.values()].sort((a, b) => a.spec.localeCompare(b.spec));
    const record = {
      schema: SCHEMA,
      recorded_utc: this.started,
      finished_utc: new Date().toISOString(),
      // Playwright's own verdict, carried rather than recomputed.
      status: result.status,
      totals: {
        passed: ran.reduce((sum, row) => sum + row.passed, 0),
        failed: ran.reduce((sum, row) => sum + row.failed, 0),
        skipped: ran.reduce((sum, row) => sum + row.skipped, 0),
      },
      // Both lists, so a single-spec run cannot look like a suite. The Python side compares them.
      spec_inventory: specs,
      specs_that_ran: ran.map(row => row.spec),
      outcomes: ran,
      spec_sha256: Object.fromEntries(
        specs.map(name => [name, sourceDigest(join(this.e2eDir, name))])),
      artefacts: listFiles(artefactDir).map(name => ({
        name,
        bytes: statSync(join(artefactDir, name)).size,
        sha256: sha256(readFileSync(join(artefactDir, name))),
      })),
      claim_boundary:
        'A record of what a rendered browser run did. Artefact digests attest that a file of '
        + 'those bytes existed when the run finished, never that a human read it and never that '
        + 'what it shows is correct. Nothing here is evidence about any scientific result.',
    };

    const target = join(this.repoRoot, 'measurements', 'browser_run.json');
    mkdirSync(join(this.repoRoot, 'measurements'), { recursive: true });
    writeFileSync(target, `${JSON.stringify(record, null, 2)}\n`, 'utf-8');
  }
}
