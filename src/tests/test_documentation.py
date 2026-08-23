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


def _routes():
    main = _read("src/api/main.py")
    return [(v.upper(), r) for v, r in
            re.findall(r'@app\.(get|post|put|delete)\("([^"]+)"', main)]


# ============================================================== coverage

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
    routes = _routes()
    claimed = re.search(r"(\w+|\d+) routes", architecture)
    assert claimed, "architecture.md should state how many routes the API serves"
    words = {"seventeen": 17, "sixteen": 16, "eighteen": 18, "fifteen": 15,
             "nineteen": 19, "twenty": 20, "fourteen": 14}
    token = claimed.group(1).lower()
    value = words.get(token, int(token) if token.isdigit() else None)
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
