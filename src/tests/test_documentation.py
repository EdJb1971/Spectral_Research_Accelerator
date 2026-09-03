"""The documentation is checked against the code, not against anyone's memory.

`architecture.md` is the stated source of truth for what is implemented, and `roadmap.md`
tracks what is planned. Both were found to have drifted: after two slices of new work the
execution-status table still claimed 152 passing tests (actual 271), Section 6 still said
"the API declares no CORS middleware" (added several slices earlier, and contradicting a
sentence two lines below it), five modules totalling ~1,800 lines had no entry in the module
assessment, and 10 of 17 API routes were unmentioned.

None of that was caught by a test, because nothing tested the documents. Prose drifts
silently in a way code does not: it keeps "passing" no matter how wrong it becomes. These
tests make the two documents fail like code.

They deliberately check *structural* facts that can be derived mechanically - does every
module have an entry, does every route appear, does the claimed test count match, is every
defect ID defined, are tasks numbered in order. They cannot check whether a description is
*true*, which is why they complement rather than replace reading it.
"""

import ast
import glob
import io
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(name: str) -> str:
    return io.open(os.path.join(REPO_ROOT, name), encoding="utf-8").read()


@pytest.fixture(scope="module")
def architecture():
    return _read("architecture.md")


@pytest.fixture(scope="module")
def roadmap():
    return _read("roadmap.md")


@pytest.fixture(scope="module")
def verification():
    return _read("VERIFICATION.md")


@pytest.fixture(scope="module")
def licence():
    return _read("LICENSE.md")


@pytest.fixture(scope="module")
def readme():
    return _read("README.md")


@pytest.fixture(scope="module")
def cross_domain():
    return _read("roadmap_cross_domain.md")


def _source_modules():
    out = []
    for path in glob.glob(os.path.join(REPO_ROOT, "src", "**", "*.py"), recursive=True):
        rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
        if "__pycache__" in rel or "/tests/" in rel:
            continue
        out.append(rel)
    return sorted(out)


def _test_files():
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(REPO_ROOT, "src", "tests", "test_*.py")))


#: Modules that define routes, and the decorator prefix each uses. `main.py` decorates `@app`
#: directly; a router module decorates `@router` and is mounted with `include_router`, and its
#: `prefix=` is prepended to every path it declares. **A new router must be
#: added here or the guard stops covering it** — which happened once already (see `_routes`) and
#: happened again in TG8.4, when `src/api/channels.py` was mounted and two real endpoints were
#: invisible until the route count disagreed with the documented one. That disagreement is the
#: only reason it was noticed, so the count claim is doing more work than it appears to.
#: Defect D75 removed the hand-maintained router list that used to live here. It named ten
#: source files, and four mounted routers were missing from it: `profiles`, `lightcurves`,
#: `ingress` and `experiment_composer`. Thirty-two real endpoints - the whole TG16 ingress
#: surface among them - were therefore invisible to every check in this file, and the count
#: claim it validated was a count of the subset the list happened to name.
#:
#: The list is gone rather than corrected. A guard whose coverage is a literal that a new
#: `include_router` call does not update is a guard that goes blind by default, and this file
#: has now done so five times (D64, D74, D75 and the two D64 records). Routes come from the
#: application object, which cannot omit a router that is mounted.


def _routes():
    """Every served route, taken from the application rather than from a list of files.

    This originally parsed `main.py` alone, then a hand-maintained tuple of router sources. Both
    forms shared one defect: adding a router did not add it here, so new endpoints were invisible
    until something else disagreed. `src/tests/test_frontend_contract.py` had already enumerated
    `app.routes` for exactly this reason, which is why *it* could see the experiment-composer
    surface while this file could not.
    """
    from src.api.main import app

    return sorted({(sorted(route.methods - {"HEAD", "OPTIONS"})[0], route.path)
                   for route in app.routes
                   if getattr(route, "path", "").startswith("/api")
                   and getattr(route, "methods", None)})


def test_every_source_module_appears_in_architecture(architecture):
    """A module nobody documented is a module nobody reviewed."""
    missing = []
    for rel in _source_modules():
        short = rel[len("src/"):]
        base = os.path.basename(rel)
        if base in ("__init__.py", "__main__.py"):
            # Packages are documented by their package section, not per file.
            pkg = os.path.dirname(short)
            if pkg and pkg + "/" in architecture:
                continue
        if short in architecture or base in architecture:
            continue
        missing.append(rel)
    assert not missing, (
        "these modules exist but are not mentioned in architecture.md:\n  "
        + "\n  ".join(missing))


def test_every_api_route_appears_in_architecture(architecture):
    """An undocumented endpoint is an untested contract."""
    missing = [r for _, r in _routes() if r not in architecture]
    assert not missing, (
        "these routes are served but not documented in architecture.md:\n  "
        + "\n  ".join(missing))


def test_route_count_claim_matches_reality(architecture):
    """Defect D74: this guard read the first "<word> routes" phrase in the whole document.

    TG17.1 wrote the words "those routes" into section 3.6zzk, 373 lines above the API surface
    heading, and from that commit the regex matched "those" — which parses as neither a numeral
    nor a spelled number, so `value` was `None` and the comparison could never be reached. The
    claim went stale by three routes and the guard reported nothing. That is the same failure as
    D64 and it is the fourth time this file has stopped covering new code silently, so the search
    is now anchored to the section that carries the claim rather than to the document.
    """
    routes = _routes()
    heading = "## 3.12 HTTP API Surface"
    assert heading in architecture, "architecture.md must carry the HTTP API surface section"
    claimed = re.search(r"(\w+|\d+) routes", architecture.split(heading, 1)[1])
    assert claimed, "the HTTP API surface section should state how many routes are served"
    words = {"seventeen": 17, "sixteen": 16, "eighteen": 18, "fifteen": 15,
             "nineteen": 19, "twenty": 20, "fourteen": 14}
    token = claimed.group(1).lower()
    value = words.get(token, int(token) if token.isdigit() else None)
    assert value is not None, (
        "the route count claim reads %r, which is neither a numeral nor a spelled number. A "
        "claim this guard cannot parse is a claim it is not checking" % token)
    assert value == len(routes), (
        "architecture.md claims %r routes but %d are served" % (token, len(routes)))


# ============================================================== counts

def test_documented_test_counts_match_the_source(architecture):
    """The count that went stale before. Counted from the AST, so it cannot be fudged."""
    actual = {}
    for name in _test_files():
        tree = ast.parse(_read("src/tests/" + name))
        actual[name] = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))

    documented = dict(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"\|\s*`(test_\w+\.py)`\s*\|\s*(\d+)\s*\|", architecture))

    assert documented, "architecture.md must carry a test inventory table (section 7.4)"

    undocumented = sorted(set(actual) - set(documented))
    assert not undocumented, (
        "test files missing from the architecture.md inventory: %s" % undocumented)

    wrong = {k: (documented[k], actual[k]) for k in documented
             if k in actual and documented[k] != actual[k]}
    assert not wrong, (
        "architecture.md test inventory is stale (documented, actual): %s" % wrong)

    stated_total = re.search(r"\|\s*\*\*total\*\*\s*\|\s*\*\*(\d+)\*\*", architecture)
    assert stated_total, "the inventory table needs a total row"
    assert int(stated_total.group(1)) == sum(actual.values()), (
        "inventory total says %s, actual %d"
        % (stated_total.group(1), sum(actual.values())))

def test_browser_suite_inventory_matches_the_spec_files(architecture):
    """The second suite, which nothing was reading.

    `frontend/e2e/` grew to the weight of a real suite while architecture.md named two of
    its files in a TG17.7 narrative and gave no current inventory at all -- so its size was
    discoverable only by listing the directory. That is the same condition section 7.4
    exists to end, reappearing one language over.

    A static reader cannot count a parametrised Playwright loop (`narrow-width.spec.ts`
    declares ten tests and collects nineteen), so this checks what it honestly can: the
    file set exactly, and each stated count as a lower bound on the literal declarations.
    The suite total stays a dated measurement in the document rather than a derivation
    here, for the same reason pytest cases are not test functions.
    """
    e2e = os.path.join(REPO_ROOT, "frontend", "e2e")
    actual = sorted(n for n in os.listdir(e2e) if n.endswith(".spec.ts"))

    documented = dict(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"\|\s*`([\w-]+\.spec\.ts)`\s*\|\s*(\d+)\s*\|", architecture))

    assert documented, (
        "architecture.md must carry a browser suite inventory (section 7.4a); the Playwright "
        "suite is not a diagnostic, it is the only check in this repository that proves a "
        "page renders")
    assert sorted(documented) == actual, (
        "browser suite inventory disagrees with frontend/e2e/ (documented, actual): %s"
        % ({"documented": sorted(documented), "actual": actual},))

    for name in actual:
        declared = len(re.findall(
            r"^\s*test\(", io.open(os.path.join(e2e, name), encoding="utf-8").read(), re.M))
        assert documented[name] >= declared, (
            "architecture.md credits %s with %d tests but the file declares %d outright; a "
            "count below the literal declarations cannot be a parametrisation"
            % (name, documented[name], declared))


# ============================================================== defect ledger

def test_every_referenced_defect_id_is_defined(architecture, roadmap, verification):
    defined = set(re.findall(r"^\| (D\d+) \|", architecture, re.M))
    assert len(defined) >= 25, "the defect ledger looks truncated: %d entries" % len(defined)
    for doc_name, doc in (("architecture.md", architecture), ("roadmap.md", roadmap),
                          ("VERIFICATION.md", verification)):
        referenced = {"D" + n for n in re.findall(r"\bD(\d+)\b", doc)}
        # Ignore anything above the ledger's range: those are not defect references
        # (for example a "D40" would be a typo worth catching, but "D2026" is a date).
        dangling = sorted(d for d in referenced - defined if int(d[1:]) <= 99)
        assert not dangling, "%s references undefined defects: %s" % (doc_name, dangling)


def test_defect_ids_are_contiguous(architecture):
    """A gap means an entry was deleted rather than resolved."""
    ids = sorted(int(d[1:]) for d in re.findall(r"^\| (D\d+) \|", architecture, re.M))
    assert ids == list(range(1, len(ids) + 1)), "defect IDs are not contiguous: %s" % ids


def test_fixed_defects_name_the_task_that_fixed_them(architecture):
    """`T3.5.24` or `T4C.5g` on the atmospheric line, `TG1.1` on the generalisation line.

    The pattern was ``T`` plus a digit until TG1.1, which is a small example of what Phase G1 is for: a
    guard that had quietly assumed the only task vocabulary there would ever be. It was found
    by the first fork task to fix a defect rather than by reading.
    """
    for line in architecture.splitlines():
        if line.startswith("| D") and "**FIXED**" in line:
            assert re.search(r"\*\*FIXED\*\*\s*TG?\d", line), (
                "a FIXED defect must name the task that fixed it: %s" % line[:80])


# ============================================================== roadmap integrity

def test_roadmap_task_numbers_are_in_order(roadmap):
    tasks = re.findall(r"^### (T3\.5\.(\d+))", roadmap, re.M)
    numbers = [int(n) for _, n in tasks]
    assert numbers == sorted(numbers), "T3.5.x sections are out of order: %s" % numbers
    assert len(numbers) == len(set(numbers)), "duplicate T3.5.x task numbers: %s" % numbers


def test_standing_rules_are_numbered_contiguously(roadmap):
    rules = [int(r[1:]) for r in re.findall(r"^### (R\d+)\.", roadmap, re.M)]
    assert rules == list(range(1, len(rules) + 1)), "R-rules are not contiguous: %s" % rules


def test_tasks_marked_done_record_their_evidence(roadmap):
    """"DONE" without evidence is the failure mode this whole exercise is about."""
    for match in re.finditer(r"^### (T3\.5\.\d+)[^\n]*\*\*DONE", roadmap, re.M):
        task = match.group(1)
        body = roadmap[match.end(): match.end() + 4000]
        body = body.split("\n### ")[0]
        assert "**Met" in body or "**Met.**" in body, (
            "%s is marked DONE but records no evidence block" % task)


#: A dated cross-domain phase declares a date and a branch, so it claims to have been
#: measured on a particular tree. The undated seam phases (TG0.x-TG2.x) predate that
#: convention and were recorded under the slice headings instead; they are exempt by
#: construction rather than by a list, because the rule keys on the date they do not carry.
#: `.` already excludes the newline, so the scan cannot run past the heading it reads.
CROSS_DOMAIN_PHASE = re.compile(
    r"^\*\*(TG\d+\.\d+).*?[—-]{1,2}\s*([A-Z][A-Z ,/]*[A-Z])\b"
    r".*?\((\d{4}-\d{2}-\d{2})[,)]", re.M)


def test_every_completed_cross_domain_phase_has_a_verification_entry(cross_domain, verification):
    """`roadmap.md` §10.2 admits no claim of completion without recorded output there.

    This guard exists because the requirement was unguarded and drifted: seven phases
    (TG17.11, TG17.12 and TG18.0 through TG18.4) reached a terminal state between
    2026-09-02 and 2026-09-04 while this file said nothing about them, and nothing in the
    suite objected. Marking a phase done in the roadmap and recording its output are two
    acts, and only the first of them was ever checked.
    """
    dated = CROSS_DOMAIN_PHASE.findall(cross_domain)
    assert len(dated) >= 19, "the phase scan looks broken: %d dated phases found" % len(dated)
    missing = sorted(phase for phase, state, _ in dated
                     if state != "IN PROGRESS" and ("## %s " % phase) not in verification)
    assert not missing, (
        "these cross-domain phases are marked complete in roadmap_cross_domain.md but have "
        "no '## <phase> ' entry in VERIFICATION.md: %s" % ", ".join(missing))


def test_a_verification_entry_is_not_written_before_the_phase_is_complete(cross_domain,
                                                                          verification):
    """The converse, and the cheaper mistake to make.

    An entry written while a phase is still in progress records output that the phase's
    remaining work can invalidate, which is the same defect as a stale calibration
    recording one gate down.
    """
    in_progress = sorted(phase for phase, state, _ in CROSS_DOMAIN_PHASE.findall(cross_domain)
                         if state == "IN PROGRESS" and ("## %s " % phase) in verification)
    assert not in_progress, (
        "VERIFICATION.md records %s, which roadmap_cross_domain.md still marks IN PROGRESS"
        % ", ".join(in_progress))


def test_no_unqualified_validated_claims(architecture, roadmap):
    """Earlier revisions claimed "validated" and "zero-error" without any evidence.

    The words are allowed, but only where the document is explicitly discussing that past
    mistake or pointing at captured output.
    """
    # Checked per *paragraph*, not per line. A line-based version failed on its own first
    # run: the disqualifying context ("Earlier revisions of this document...") sat on the
    # preceding line, so a sentence explicitly disowning the claim was flagged as making it.
    for name, doc in (("architecture.md", architecture), ("roadmap.md", roadmap)):
        for para in doc.split("\n\n"):
            low = " ".join(para.lower().split())
            if "zero-error" in low or "fully validated" in low:
                assert any(k in low for k in ("earlier revision", "aspirational",
                                              "not substantiated", "claimed", "it was not")), (
                    "%s makes an unqualified validation claim: %s" % (name, para[:160]))


# ============================================================== benchmark suite

def test_benchmark_status_in_docs_matches_a_real_run(architecture):
    """The document states a PASS/FAIL/PENDING triple; it must be the true one."""
    from src.benchmarks import run_all
    from src.benchmarks.runner import summarise

    counts = summarise(run_all())
    claim = re.search(r"\*\*(\d+) PASS, (\d+) FAIL, (\d+) NOT_YET_RUNNABLE\*\*", architecture)
    assert claim, "architecture.md should state the benchmark suite's status"
    assert (int(claim.group(1)), int(claim.group(2)), int(claim.group(3))) == (
        counts["PASS"], counts["FAIL"], counts["NOT_YET_RUNNABLE"]), (
        "architecture.md claims %s PASS / %s FAIL / %s pending, actual %d / %d / %d"
        % (claim.group(1), claim.group(2), claim.group(3),
           counts["PASS"], counts["FAIL"], counts["NOT_YET_RUNNABLE"]))


def test_every_benchmark_is_named_in_the_docs(architecture, verification):
    """A gate that exists but is undocumented cannot be relied on by a reader."""
    from src.benchmarks import all_benchmarks
    combined = architecture + verification
    missing = [b.name for b in all_benchmarks() if b.name not in combined]
    assert not missing, "benchmarks absent from the docs: %s" % missing


# ============================================================== self-consistency

def test_architecture_does_not_contradict_itself_about_cors(architecture):
    """The specific contradiction this file was written after finding."""
    assert "declares no CORS middleware" not in architecture
    assert "CORSMiddleware" in architecture


def test_architecture_does_not_claim_the_frontend_was_never_built(architecture):
    assert "has never been installed or built" not in architecture
    # ...but the genuinely unverified part must still be stated.
    assert "rendered appearance" in architecture.lower()


def test_proprietary_licence_preserves_owner_and_named_researcher_boundary(licence):
    """The intended family grant must not silently become all-rights-reserved or open source."""
    required = (
        "Edward Jonathan Bentley", "ed.j.bentley@gmail.com",
        "Adam Frank Bentley", "adam.f.bentley@gmail.com",
        "perpetual", "worldwide", "royalty-free", "commercial activity",
        "high-performance-computing", "Independent Extension",
        "must not", "publicly distribute", "sublicensed",
        "Third-party materials", "laws of New Zealand",
    )
    for text in required:
        assert text in licence, "LICENSE.md has lost the declared term %r" % text
    assert "not an open-source" in licence
    assert "does not assign or transfer ownership" in licence

# ============================================================== status-section drift

def _claimed_test_count(doc):
    match = re.search(r"Backend test suite \| \*\*(\d+) passed, (\d+) xfailed", doc)
    assert match, "the status table must state the passing test count"
    return int(match.group(1)), int(match.group(2))


def test_status_sections_agree_on_the_test_count(architecture, roadmap):
    """The two status tables must not disagree with each other.

    This guard was added because `roadmap.md` Section 1 was found claiming **271 passed**
    and "DTCWT still not implemented as advertised" several slices after both had changed,
    while `architecture.md` said 446. Neither number was current and nothing failed. The
    exact pytest total cannot be derived without running pytest, but two documents
    disagreeing about it is decisive evidence that at least one is stale - and it is the
    cheap half of the check that would have caught this.
    """
    assert _claimed_test_count(architecture) == _claimed_test_count(roadmap), (
        "architecture.md claims %s and roadmap.md claims %s"
        % (_claimed_test_count(architecture), _claimed_test_count(roadmap)))


def test_claimed_test_count_is_at_least_the_function_count(architecture):
    """A necessary condition that is checkable statically: parametrisation only adds cases."""
    claimed, _ = _claimed_test_count(architecture)
    total = 0
    for name in _test_files():
        tree = ast.parse(_read("src/tests/" + name))
        total += sum(1 for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    assert claimed >= total, (
        "the status table claims %d passing tests but %d test functions exist; pytest "
        "cannot report fewer cases than there are functions unless tests are failing, "
        "skipped or uncollected" % (claimed, total))


def test_status_sections_agree_on_the_defect_ledger(architecture, roadmap):
    """roadmap.md summarises the ledger; the summary must match the ledger itself."""
    rows = re.findall(r"^\| (D\d+) \|(.*)$", architecture, re.M)
    fixed = [d for d, body in rows if "**FIXED**" in body]
    partial = [d for d, body in rows if "PARTIAL" in body]
    open_ids = [d for d, body in rows
                if "**FIXED**" not in body and "PARTIAL" not in body]

    # The id lists are plural: the guard was first written when exactly one defect was open,
    # and a regex that only matches one forces the *document* to be wrong whenever a second
    # one is found - which is precisely backwards for a consistency check.
    claim = re.search(
        r"\(D1-D(\d+),\s+of which\s+\*\*(\d+) fixed,\s+(\d+) partial \(([^)]+)\),\s+"
        r"(\d+) open \(([^)]+)\)\*\*\)", roadmap)
    assert claim, ("roadmap.md Section 1 must state the ledger range and counts in the "
                   "form '(D1-D41, of which **38 fixed, 1 partial (D18), 2 open (D17, D41)**)'")

    def ids(text):
        return [part.strip() for part in text.split(",") if part.strip()]

    assert int(claim.group(1)) == len(rows)
    assert int(claim.group(2)) == len(fixed)
    assert int(claim.group(3)) == len(partial)
    assert ids(claim.group(4)) == partial
    assert int(claim.group(5)) == len(open_ids)
    assert ids(claim.group(6)) == open_ids


def test_every_registered_tg17_receipt_capability_has_a_documented_explanation(architecture):
    """The Platform trust surface and source-of-truth document are one generated contract.

    An operation, adapter, refusal or export field added without prose would otherwise be usable
    before a scientist could learn what it means. Backticks make this an identity check rather
    than an accidental substring match.
    """
    from src.core.experiment_receipt import capability_snapshot

    snapshot = capability_snapshot()
    names = ([row["name"] for row in snapshot["operations"]]
             + [row["adapter_id"] for row in snapshot["adapters"]]
             + [row["name"] for row in snapshot["refusals"]]
             + [row["name"] for row in snapshot["receipt_fields"]])
    missing = sorted(name for name in names if "`%s`" % name not in architecture)
    assert not missing, (
        "registered TG17 receipt capabilities have no visible architecture explanation: %s"
        % missing)


# ---------------------------------------------------------------------------
# The two documents nothing was reading.
#
# `README.md` and `roadmap_cross_domain.md` were unguarded for the whole
# programme, and both drifted exactly as the guarded ones had before these
# tests existed. At the T4E.2 audit the README still said "the decisive Phase
# 4C real-ERA5 gate has not been run", "the 8,764-frame record has not been
# acquired and no gate has run" and "D43 remains open" -- several slices after
# T4C.5m acquired 8,764 frames, ran the gate to a PASS and closed D43 -- while
# `architecture.md`'s ledger prose still summarised 78 entries and 75 fixes
# against a table holding 90 and 87. Nothing failed, because nothing looked.
# ---------------------------------------------------------------------------

def _ledger_open_ids(architecture):
    rows = re.findall(r"^\| (D\d+) \|(.*)$", architecture, re.M)
    return [d for d, body in rows
            if "**FIXED**" not in body and "PARTIAL" not in body]


def _ledger_fixed_ids(architecture):
    rows = re.findall(r"^\| (D\d+) \|(.*)$", architecture, re.M)
    return [d for d, body in rows if "**FIXED**" in body]


def _readme_frontier(readme):
    start = readme.find("**Current frontier")
    assert start != -1, (
        "README.md must carry a '**Current frontier' block naming where the programme is; "
        "an orientation document with no 'you are here' sends a reader to guess from the "
        "feature list, which is how this file came to claim an unrun gate for four slices")
    end = readme.find("**What has not been done**", start)
    assert end != -1, "the frontier block must be followed by the '**What has not been done**' list"
    return readme[start:end]


def test_readme_disclaims_being_the_status_of_record(readme):
    """The README is read first and updated last; it must say which document to trust."""
    assert "not** the status of record" in readme or "not the status of record" in readme, (
        "README.md must state that it is not the status of record")
    for name in ("architecture.md", "roadmap.md", "roadmap_cross_domain.md", "VERIFICATION.md"):
        assert name in readme, "README.md must point at %s in its document map" % name


def test_readme_open_defects_match_the_ledger(readme, architecture):
    """The README's frontier block lists the open defects; the ledger decides them."""
    block = _readme_frontier(readme)
    match = re.search(r"Open defects: \*\*([^*]+)\*\*", block)
    assert match, ("the README frontier block must state 'Open defects: **D84 and D85**' or "
                   "equivalent")
    claimed = sorted(re.findall(r"D\d+", match.group(1)))
    assert claimed == sorted(_ledger_open_ids(architecture)), (
        "README.md claims open defects %s but the architecture.md ledger holds %s"
        % (claimed, sorted(_ledger_open_ids(architecture))))


def test_readme_test_count_agrees_with_the_status_tables(readme, architecture):
    """A third document quoting the suite total is a third place for it to go stale."""
    match = re.search(r"full backend run: \*\*(\d+) passed", readme)
    assert match, "the README frontier block must quote the last measured full-suite total"
    assert int(match.group(1)) == _claimed_test_count(architecture)[0], (
        "README.md claims %s passing tests, architecture.md claims %s"
        % (match.group(1), _claimed_test_count(architecture)[0]))


def test_readme_frontier_task_is_not_already_done(readme, roadmap, cross_domain):
    """The frontier moves every slice; a README naming a finished task as 'next' is stale."""
    block = _readme_frontier(readme)
    for task in re.findall(r"\*\*(T[G]?[0-9][A-Z]?\.[0-9]+(?:\.[0-9]+)?) [^*]*is next", block):
        for doc in (roadmap, cross_domain):
            for heading in re.finditer(r"\*\*%s [A-Z]" % re.escape(task), doc):
                window = doc[max(0, heading.start() - 40): heading.start() + 200]
                assert "DONE" not in window, (
                    "README.md calls %s the next task but the roadmap marks it DONE" % task)


def test_no_current_status_section_calls_a_fixed_defect_open(readme, roadmap, architecture):
    """Historical task prose may say 'D43 remains open'; a *current status* section may not.

    The regions checked are the ones a reader treats as present tense: the README's
    orientation section, `roadmap.md` Section 1, and the ledger summary paragraph in
    `architecture.md`. Each is bounded, so slice narratives recording what was true at the
    time keep saying so.
    """
    fixed = set(_ledger_fixed_ids(architecture))
    regions = {
        "README.md orientation": readme[:readme.find("Key Scientific Pillars")],
        "roadmap.md Section 1": roadmap[roadmap.find("## 1. Honest Technical Status"):
                                        roadmap.find("## 2.")],
        "architecture.md ledger summary": architecture[
            architecture.find("The ledger below has grown"):
            architecture.find("The ledger below has grown") + 900],
    }
    pattern = re.compile(r"\*{0,2}(D\d+)\*{0,2}[^.\n]{0,80}?(remains open|is still open|"
                         r"still open|remains unfixed)")
    for where, text in regions.items():
        assert text, "could not locate the %s region" % where
        for defect, phrase in pattern.findall(text):
            assert defect not in fixed, (
                "%s says %s %s, but the ledger marks it FIXED" % (where, defect, phrase))


def test_cross_domain_rules_continue_contiguously_from_r17(roadmap, cross_domain):
    """R1-R16 live in roadmap.md and R17 onward in roadmap_cross_domain.md, by declaration.

    Two files holding one numbering is a gap or a collision waiting to happen, and a rule that
    exists in neither reachable range is a lesson nobody will find.
    """
    atmospheric = [int(r[1:]) for r in re.findall(r"^### (R\d+)\.", roadmap, re.M)]
    extended = [int(r[1:]) for r in re.findall(r"^### (R\d+)\.", cross_domain, re.M)]
    assert extended == list(range(max(atmospheric) + 1, max(atmospheric) + 1 + len(extended))), (
        "roadmap.md ends at R%d, so roadmap_cross_domain.md must run R%d upward; it holds %s"
        % (max(atmospheric), max(atmospheric) + 1, extended))
    assert not set(atmospheric) & set(extended), "the two rule ranges overlap"
    assert ("continues at **R17**" in roadmap or "continues at R17" in roadmap), (
        "roadmap.md must say where the rule numbering continues")
