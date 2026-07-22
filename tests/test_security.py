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


def test_detects_hardcoded_secret():
    diff = make_diff("config.py", ['API_KEY = "sk-abc123def456"'])
    findings = scan_security(diff)
    assert any(f["category"] == "hardcoded-secret" and f["severity"] == "high" for f in findings)


def test_detects_sql_concatenation():
    diff = make_diff("db.py", ["query = \"SELECT * FROM users WHERE name = '\" + user + \"'\""])
    findings = scan_security(diff)
    assert any(f["category"] == "injection" for f in findings)


def test_detects_eval():
    diff = make_diff("util.py", ["result = eval(user_input)"])
    findings = scan_security(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)


def test_detects_disabled_tls_verification():
    diff = make_diff("client.py", ["requests.get(url, verify=False)"])
    findings = scan_security(diff)
    assert any(f["category"] == "insecure-transport" for f in findings)


def test_detects_weak_hash():
    diff = make_diff("hashing.py", ["digest = hashlib.md5(data).hexdigest()"])
    findings = scan_security(diff)
    assert any(f["category"] == "weak-crypto" for f in findings)


def test_flags_sensitive_filename():
    diff = make_diff("auth.py", ["x = 1"])
    findings = scan_security(diff)
    assert any(f["category"] == "sensitive-file" and f["file"] == "auth.py" for f in findings)


def test_flags_workflow_file():
    diff = make_diff(".github/workflows/deploy.yml", ["run: echo ok"])
    findings = scan_security(diff)
    assert any(f["category"] == "sensitive-file" for f in findings)


def test_clean_diff_has_no_findings():
    diff = make_diff("math_helpers.py", ["total = a + b", "return total"])
    findings = scan_security(diff)
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
    findings = scan_security(diff)
    assert findings == []


def test_finding_reports_correct_line_number():
    diff = make_diff("script.py", ["safe_line = 1", "os.system(cmd)"])
    findings = scan_security(diff)
    dangerous = [f for f in findings if f["category"] == "dangerous-call"]
    assert len(dangerous) == 1
    # hunk starts at new line 1: context line is 1, added lines are 2 and 3
    assert dangerous[0]["line"] == 3


def test_findings_sorted_high_before_medium():
    diff = make_diff("mixed.py", [
        "digest = hashlib.md5(data).hexdigest()",
        'PASSWORD = "hunter2secret"',
    ])
    findings = scan_security(diff)
    code = [f for f in findings if f["category"] != "sensitive-file"]
    assert code[0]["severity"] == "high"
    assert code[-1]["severity"] == "medium"


def test_report_lists_findings_with_location():
    diff = make_diff("db.py", ["result = eval(user_input)"])
    report = format_security_report(scan_security(diff))
    assert "## Security Findings (heuristic scan)" in report
    assert "db.py:2" in report
    assert "dangerous-call" in report


def test_report_when_clean():
    report = format_security_report([])
    assert "None detected" in report


def test_report_separates_sensitive_files_section():
    diff = make_diff("login.py", ["x = 1"])
    report = format_security_report(scan_security(diff))
    assert "## Security-Sensitive Areas Touched" in report
    assert "login.py" in report


def test_detects_unquoted_env_secret():
    diff = make_diff(".env.production", ["STRIPE_SECRET_KEY=sk_live_a8f3k2j9d0s1"])
    findings = scan_security(diff)
    assert any(f["category"] == "hardcoded-secret" and f["severity"] == "high" for f in findings)


def test_detects_json_api_key():
    diff = make_diff("config.json", ['    "api_key": "a8f3k2j9d0s1x7c4",'])
    findings = scan_security(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_detects_php_sql_interpolation():
    diff = make_diff("query.php", ['$q = "SELECT * FROM users WHERE id = $id";'])
    findings = scan_security(diff)
    assert any(f["category"] == "injection" for f in findings)


def test_detects_innerhtml_xss_sink():
    diff = make_diff("render.js", ["element.innerHTML = userComment;"])
    findings = scan_security(diff)
    assert any(f["category"] == "xss" for f in findings)


def test_detects_actions_expression_injection():
    diff = make_diff(".github/workflows/build.yml",
                     ['      run: echo "${{ github.event.pull_request.title }}"'])
    findings = scan_security(diff)
    assert any(f["category"] == "workflow-injection" and f["severity"] == "high" for f in findings)


def test_placeholder_value_is_not_flagged_as_secret():
    diff = make_diff("config.py", ['API_KEY = "your-api-key"'])
    findings = scan_security(diff)
    assert not any(f["category"] == "hardcoded-secret" for f in findings)


def test_secret_evidence_is_redacted():
    diff = make_diff("config.py", ['PASSWORD = "hunter2realvalue"'])
    findings = scan_security(diff)
    secret = [f for f in findings if f["category"] == "hardcoded-secret"][0]
    assert "hunter2realvalue" not in secret["evidence"]
    assert "[REDACTED]" in secret["evidence"]


def test_secret_without_separator_is_fully_redacted():
    diff = make_diff("notes.txt", ["aws key AKIA1234567890ABCDEF"])
    findings = scan_security(diff)
    secret = [f for f in findings if f["category"] == "hardcoded-secret"][0]
    assert "AKIA1234567890ABCDEF" not in secret["evidence"]


def test_one_finding_per_category_per_line():
    # matches both the quoted-credential and AWS-key secret rules
    diff = make_diff("deploy.py", ['AWS_KEY = "AKIA1234567890ABCDEF"'])
    findings = scan_security(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_lockfiles_are_still_scanned():
    diff = make_diff("package-lock.json", ['"npm_token": "a8f3k2j9d0s1x7c4"'])
    findings = scan_security(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_placeholder_word_in_trailing_comment_does_not_suppress_real_secret():
    """Regression: a real secret followed by '# example' must still be flagged.
    Placeholder detection must only look at the matched credential text,
    not the whole line."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+API_KEY = "sk-live-abc123realvalue" # example\n'
    )
    findings = scan_security(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_unquoted_secret_with_inline_comment_is_detected():
    """Regression: an unquoted credential followed by an inline comment
    must still match, not silently pass through."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        "+TOKEN = abc123realtoken456  # inline comment\n"
    )
    findings = scan_security(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_secret_redaction_applies_to_every_finding_on_the_same_line():
    """Regression: when a line has both a hardcoded secret AND another
    issue (e.g. eval), the secret must not leak via the OTHER finding's
    evidence."""
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+PASSWORD = "real-secret-value"; eval(user_input)\n'
    )
    findings = scan_security(diff)
    for f in findings:
        assert "real-secret-value" not in f["evidence"]
