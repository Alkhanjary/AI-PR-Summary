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
