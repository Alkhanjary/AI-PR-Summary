import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarize_pr import scan_security, format_security_report


def make_diff(filename, added_lines):
    added = "".join(f"+{line}\n" for line in added_lines)
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"index 1111111..2222222 100644\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -1,2 +1,{2 + len(added_lines)} @@\n"
        f" def existing():\n"
        f"{added}"
        f"     pass\n"
    )


def scan(diff_text):
    """Convenience wrapper: most tests only care about the gating findings."""
    findings, _suppressed = scan_security(diff_text)
    return findings


def test_detects_hardcoded_secret():
    diff = make_diff("config.py", ['API_KEY = "sk-abc123def456"'])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" and f["severity"] == "high" for f in findings)


def test_detects_sql_concatenation():
    diff = make_diff("db.py", ["query = \"SELECT * FROM users WHERE name = '\" + user + \"'\""])
    findings = scan(diff)
    assert any(f["category"] == "injection" for f in findings)


def test_detects_eval():
    diff = make_diff("util.py", ["result = eval(user_input)"])
    findings = scan(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)


def test_detects_disabled_tls_verification():
    diff = make_diff("client.py", ["requests.get(url, verify=False)"])
    findings = scan(diff)
    assert any(f["category"] == "insecure-transport" for f in findings)


def test_detects_weak_hash():
    diff = make_diff("hashing.py", ["digest = hashlib.md5(data).hexdigest()"])
    findings = scan(diff)
    assert any(f["category"] == "weak-crypto" for f in findings)


def test_flags_sensitive_filename():
    diff = make_diff("auth.py", ["x = 1"])
    findings = scan(diff)
    assert any(f["category"] == "sensitive-file" and f["file"] == "auth.py" for f in findings)


def test_flags_workflow_file():
    diff = make_diff(".github/workflows/deploy.yml", ["run: echo ok"])
    findings = scan(diff)
    assert any(f["category"] == "sensitive-file" for f in findings)


def test_clean_diff_has_no_findings():
    diff = make_diff("math_helpers.py", ["total = a + b", "return total"])
    findings = scan(diff)
    assert findings == []


def test_removed_lines_are_not_scanned():
    diff = (
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,1 @@\n"
        "-result = eval(user_input)\n"
        " pass\n"
    )
    findings = scan(diff)
    assert findings == []


def test_finding_reports_correct_line_number():
    diff = make_diff("script.py", ["safe_line = 1", "os.system(cmd)"])
    findings = scan(diff)
    dangerous = [f for f in findings if f["category"] == "dangerous-call"]
    assert len(dangerous) == 1
    # hunk starts at new line 1: context line is 1, added lines are 2 and 3
    assert dangerous[0]["line"] == 3


def test_findings_sorted_high_before_medium():
    diff = make_diff("mixed.py", [
        "digest = hashlib.md5(data).hexdigest()",
        'PASSWORD = "hunter2secret"',
    ])
    findings = scan(diff)
    code = [f for f in findings if f["category"] != "sensitive-file"]
    assert code[0]["severity"] == "high"
    assert code[-1]["severity"] == "medium"


def test_report_lists_findings_with_location():
    diff = make_diff("db.py", ["result = eval(user_input)"])
    report = format_security_report(scan(diff))
    assert "## Security Findings (heuristic scan)" in report
    assert "db.py:2" in report
    assert "dangerous-call" in report


def test_report_when_clean():
    report = format_security_report([])
    assert "None detected" in report


def test_report_separates_sensitive_files_section():
    diff = make_diff("login.py", ["x = 1"])
    report = format_security_report(scan(diff))
    assert "## Security-Sensitive Areas Touched" in report
    assert "login.py" in report


def test_detects_unquoted_env_secret():
    diff = make_diff(".env.production", ["STRIPE_SECRET_KEY=sk_live_a8f3k2j9d0s1"])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" and f["severity"] == "high" for f in findings)


def test_detects_json_api_key():
    diff = make_diff("config.json", ['    "api_key": "a8f3k2j9d0s1x7c4",'])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_detects_php_sql_interpolation():
    diff = make_diff("query.php", ['$q = "SELECT * FROM users WHERE id = $id";'])
    findings = scan(diff)
    assert any(f["category"] == "injection" for f in findings)


def test_detects_innerhtml_xss_sink():
    diff = make_diff("render.js", ["element.innerHTML = userComment;"])
    findings = scan(diff)
    assert any(f["category"] == "xss" for f in findings)


def test_detects_actions_expression_injection():
    diff = make_diff(".github/workflows/build.yml",
                     ['      run: echo "${{ github.event.pull_request.title }}"'])
    findings = scan(diff)
    assert any(f["category"] == "workflow-injection" and f["severity"] == "high" for f in findings)


def test_placeholder_value_is_not_flagged_as_secret():
    diff = make_diff("config.py", ['API_KEY = "your-api-key"'])
    findings = scan(diff)
    assert not any(f["category"] == "hardcoded-secret" for f in findings)


def test_secret_evidence_is_redacted():
    diff = make_diff("config.py", ['PASSWORD = "hunter2realvalue"'])
    findings = scan(diff)
    secret = [f for f in findings if f["category"] == "hardcoded-secret"][0]
    assert "hunter2realvalue" not in secret["evidence"]
    assert "[REDACTED]" in secret["evidence"]


def test_secret_without_separator_is_fully_redacted():
    diff = make_diff("notes.txt", ["aws key AKIA1234567890ABCDEF"])
    findings = scan(diff)
    secret = [f for f in findings if f["category"] == "hardcoded-secret"][0]
    assert "AKIA1234567890ABCDEF" not in secret["evidence"]


def test_one_finding_per_category_per_line():
    # matches both the quoted-credential and AWS-key secret rules
    diff = make_diff("deploy.py", ['AWS_KEY = "AKIA1234567890ABCDEF"'])
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_lockfiles_are_still_scanned():
    diff = make_diff("package-lock.json", ['"npm_token": "a8f3k2j9d0s1x7c4"'])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_secret_redaction_applies_to_every_finding_on_the_same_line():
    """Regression: when a line has both a hardcoded secret AND another
    issue (e.g. eval), the secret must not leak via the OTHER finding's
    evidence."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+PASSWORD = "real-secret-value"; eval(user_input)\n'
    )
    findings = scan(diff)
    for f in findings:
        assert "real-secret-value" not in f["evidence"]


# --- Placeholder classification must look only at the credential VALUE ---
# (Regression for a confirmed bypass: checking the whole regex match let a
# placeholder-ish variable NAME or a trailing comment hide a real secret.)

def test_placeholder_word_in_variable_name_does_not_suppress_real_secret():
    """A variable named EXAMPLE_API_KEY with a REAL value must still be
    flagged — placeholder detection must look at the value, not the name."""
    diff = make_diff("config.py", ['EXAMPLE_API_KEY = "sk-live-realvalue123"'])
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_placeholder_word_in_trailing_comment_does_not_suppress_real_secret():
    """A real secret followed by '# example' must still be flagged."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+API_KEY = "sk-live-abc123realvalue" # example\n'
    )
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_unquoted_secret_with_placeholder_word_after_comment_is_detected():
    """Regression for the exact reviewer-reported bypass: an unquoted
    credential followed by '# example' must still match."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        "+TOKEN=abc123realtoken456 # example\n"
    )
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_unquoted_secret_with_inline_comment_is_detected():
    """An unquoted credential followed by an inline comment must still
    match, not silently pass through."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        "+TOKEN = abc123realtoken456  # inline comment\n"
    )
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_genuine_placeholder_value_still_suppressed():
    """A real placeholder VALUE (not just a placeholder-ish name) is still
    correctly suppressed."""
    diff = make_diff("config.py", ['REAL_LOOKING_NAME = "your-api-key"'])
    findings = scan(diff)
    assert not any(f["category"] == "hardcoded-secret" for f in findings)


# --- "# nosec" suppression: visible waiver, not a silent bypass ---
#
# The diff (and therefore any inline "# nosec" comment) is fully
# attacker-controlled. A high-confidence category must never be waved away
# by a comment on the same attacker-supplied line. Lower-confidence
# categories (prone to false positives, e.g. XSS-sink or weak-crypto pattern
# definitions inside this very file) may be suppressed from the merge gate,
# but the suppressed finding is always still reported for human review.

def test_nosec_does_not_suppress_hardcoded_secret():
    """High-confidence categories ignore '# nosec' entirely — this is the
    exact reviewer-reported bypass (secret waved away by an inline comment)."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+API_KEY = "sk-live-realvalue123"  # nosec\n'
    )
    findings, suppressed = scan_security(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)
    assert suppressed == []


def test_nosec_does_not_suppress_dangerous_call():
    diff = (
        "diff --git a/runner.py b/runner.py\n"
        "@@ -0,0 +1 @@\n"
        "+result = eval(user_input)  # nosec\n"
    )
    findings, suppressed = scan_security(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)
    assert suppressed == []


def test_nosec_does_not_suppress_injection():
    diff = (
        "diff --git a/db.py b/db.py\n"
        "@@ -0,0 +1 @@\n"
        '+query = "SELECT * FROM users WHERE name = \'" + user + "\'"  # nosec\n'
    )
    findings, suppressed = scan_security(diff)
    assert any(f["category"] == "injection" for f in findings)
    assert suppressed == []


def test_nosec_does_not_suppress_workflow_injection():
    diff = make_diff(".github/workflows/build.yml",
                     ['      run: echo "${{ github.event.pull_request.title }}"  # nosec'])
    findings, suppressed = scan_security(diff)
    assert any(f["category"] == "workflow-injection" for f in findings)
    assert suppressed == []


def test_nosec_suppresses_low_confidence_category_but_stays_visible():
    """A suppressible category (e.g. xss) moves out of 'findings' (the gate)
    into 'suppressed', but is never silently dropped from the report."""
    diff = make_diff("render.js", ["element.innerHTML = trustedTemplate;  # nosec"])
    findings, suppressed = scan_security(diff)
    assert not any(f["category"] == "xss" for f in findings)
    assert any(f["category"] == "xss" for f in suppressed)


def test_nosec_marker_only_affects_the_marked_line():
    diff = make_diff("render.js", [
        "element.innerHTML = trustedTemplate;  # nosec",
        "other.innerHTML = untrustedInput;",
    ])
    findings, suppressed = scan_security(diff)
    assert any(f["category"] == "xss" for f in findings)
    assert any(f["category"] == "xss" for f in suppressed)


def test_suppressed_findings_appear_in_report():
    diff = make_diff("render.js", ["element.innerHTML = trustedTemplate;  # nosec"])
    findings, suppressed = scan_security(diff)
    report = format_security_report(findings, suppressed)
    assert "## Suppressed by # nosec (review before trusting)" in report
    assert "xss" in report


# --- tests/docs paths must still be scanned: no exclusion by path ---
# (A path-based exclusion is itself an attacker-controlled bypass surface:
# an attacker can put a real payload in any file whose path merely contains
# "tests/" or "docs/". This is the exact reviewer-reported bypass.)

def test_dangerous_code_under_tests_path_is_still_detected():
    diff = make_diff("tests/exploit.py", ["import os; os.system(attacker_cmd)"])
    findings = scan(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)


def test_secret_under_tests_path_is_still_detected():
    """Exact reviewer-reported bypass: a real credential added to
    tests/test_deploy.py must still be flagged — tests execute in CI, and
    secrets in test/doc files are still leaked secrets."""
    diff = make_diff("tests/test_deploy.py", ['API_KEY = "sk-live-realvalue123"'])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_secret_under_docs_path_is_still_detected():
    diff = make_diff("docs/setup.py", ['API_KEY = "sk-live-realvalue123"'])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_dangerous_code_in_markdown_file_is_still_detected():
    diff = make_diff("README.md", ["result = eval(user_input)"])
    findings = scan(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)
