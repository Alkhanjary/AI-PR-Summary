import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarize_pr import (
    scan_security,
    format_security_report,
    redact_secret,
    redact_diff_for_llm,
    compute_fingerprint,
    load_baseline,
)


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


def scan(diff_text, baseline=None):
    """Convenience wrapper: most tests only care about the gating findings."""
    findings, _baselined = scan_security(diff_text, baseline=baseline)
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


def test_placeholder_word_in_variable_name_does_not_suppress_real_secret():
    """A variable named EXAMPLE_API_KEY with a REAL value must still be
    flagged — placeholder detection must look at the value, not the name."""
    diff = make_diff("config.py", ['EXAMPLE_API_KEY = "sk-live-realvalue123"'])
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


# --- Placeholder detection must fullmatch the VALUE, not substring-search it ---
# (Regression: "sk-live-example-real123" contains the substring "example" but
# is NOT a placeholder value as a whole, and must still be flagged.)

def test_placeholder_substring_inside_real_value_does_not_suppress():
    diff = make_diff("config.py", ['API_KEY = "sk-live-example-real123"'])
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_placeholder_word_in_trailing_comment_does_not_suppress_real_secret():
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+API_KEY = "sk-live-abc123realvalue" # example\n'
    )
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


def test_unquoted_secret_with_placeholder_word_after_comment_is_detected():
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        "+TOKEN=abc123realtoken456 # example\n"
    )
    findings = scan(diff)
    secrets = [f for f in findings if f["category"] == "hardcoded-secret"]
    assert len(secrets) == 1


# --- Case sensitivity: unquoted assignments must match any variable case ---

def test_lowercase_unquoted_assignment_is_detected():
    diff = make_diff("config.py", ["token=abc123realsecret456"])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_mixed_case_unquoted_assignment_is_detected():
    diff = make_diff("config.py", ["Api_Key=abc123realsecret456"])
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


# --- "+++" line-header collision: an added line starting with "++" must
# still be scanned as code, not mistaken for the diff's own file header ---

def test_added_line_starting_with_double_plus_is_still_scanned():
    # The diff marker "+" plus content "++eval(user_input)" produces the raw
    # line "+++eval(user_input)", which must NOT be treated as a "+++ b/file"
    # header just because it starts with "+++".
    diff = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,2 @@\n"
        " context\n"
        "+++eval(user_input)\n"
    )
    findings = scan(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)


def test_real_file_header_lines_are_still_ignored():
    # Sanity check the fix doesn't regress the normal case: the genuine
    # "+++ b/file" header (before any hunk) must not itself be scanned.
    diff = make_diff("safe.py", ["total = a + b"])
    findings = scan(diff)
    assert findings == []


# --- redact_secret: must redact the exact matched span, not split on the
# first ':'/'=' in the line (which can retain the secret itself) ---

def test_redact_secret_by_span_hides_value_preceding_separator():
    content = "AKIA1234567890ABCDEF: this is my prod key"
    redacted = redact_secret(content, span=(0, len("AKIA1234567890ABCDEF")))
    assert "AKIA1234567890ABCDEF" not in redacted
    assert "[REDACTED]" in redacted


def test_scan_does_not_leak_aws_key_preceding_colon():
    diff = make_diff("notes.txt", ["AKIA1234567890ABCDEF: this is my prod key"])
    findings = scan(diff)
    secret = [f for f in findings if f["category"] == "hardcoded-secret"][0]
    assert "AKIA1234567890ABCDEF" not in secret["evidence"]


# --- "# nosec" is diff content (attacker-controlled): it must be purely
# informational now, and must NEVER remove a finding from the gate ---

def test_nosec_does_not_remove_hardcoded_secret_from_findings():
    diff = (
        "diff --git a/config.py b/config.py\n"
        "@@ -0,0 +1 @@\n"
        '+API_KEY = "sk-live-realvalue123"  # nosec\n'
    )
    findings = scan(diff)
    assert any(f["category"] == "hardcoded-secret" for f in findings)


def test_nosec_does_not_remove_medium_severity_finding_from_findings():
    """Exact reviewer-reported bypass: verify=False # nosec must still gate
    at --fail-on medium (i.e. must still be in `findings`)."""
    diff = make_diff("client.py", ["requests.get(url, verify=False)  # nosec"])
    findings = scan(diff)
    assert any(f["category"] == "insecure-transport" for f in findings)


def test_nosec_marks_finding_as_acknowledged_but_keeps_it_gated():
    diff = make_diff("client.py", ["requests.get(url, verify=False)  # nosec"])
    findings = scan(diff)
    finding = [f for f in findings if f["category"] == "insecure-transport"][0]
    assert finding["acknowledged"] is True


def test_without_nosec_finding_is_not_acknowledged():
    diff = make_diff("client.py", ["requests.get(url, verify=False)"])
    findings = scan(diff)
    finding = [f for f in findings if f["category"] == "insecure-transport"][0]
    assert finding["acknowledged"] is False


def test_acknowledged_finding_is_noted_in_report_but_still_listed():
    diff = make_diff("client.py", ["requests.get(url, verify=False)  # nosec"])
    report = format_security_report(scan(diff))
    assert "nosec" in report
    assert "insecure-transport" in report


# --- tests/docs paths must still be scanned: no exclusion by path ---

def test_dangerous_code_under_tests_path_is_still_detected():
    diff = make_diff("tests/exploit.py", ["import os; os.system(attacker_cmd)"])
    findings = scan(diff)
    assert any(f["category"] == "dangerous-call" for f in findings)


def test_secret_under_tests_path_is_still_detected():
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


# --- Trusted baseline: the only mechanism that can actually remove a
# finding from the gate. Must be an explicit fingerprint match, not a
# blanket file/category exemption, and must respect expiry. ---

def test_baseline_excludes_matching_finding_from_gate():
    diff = make_diff("README.md", ["result = eval(user_input)"])
    unbaselined_findings = scan(diff)
    fp = compute_fingerprint(unbaselined_findings[0])
    baseline = {fp: {"owner": "alice", "reason": "doc example", "expires": "2099-01-01"}}

    findings, baselined = scan_security(diff, baseline=baseline)
    assert findings == []
    assert len(baselined) == 1
    assert baselined[0]["baseline_owner"] == "alice"


def test_baseline_entry_still_appears_in_report():
    diff = make_diff("README.md", ["result = eval(user_input)"])
    unbaselined_findings = scan(diff)
    fp = compute_fingerprint(unbaselined_findings[0])
    baseline = {fp: {"owner": "alice", "reason": "doc example", "expires": "2099-01-01"}}

    findings, baselined = scan_security(diff, baseline=baseline)
    report = format_security_report(findings, baselined)
    assert "## Baselined" in report
    assert "alice" in report
    assert "doc example" in report


def test_expired_baseline_entry_does_not_exclude_finding(tmp_path):
    diff = make_diff("README.md", ["result = eval(user_input)"])
    unbaselined_findings = scan(diff)
    fp = compute_fingerprint(unbaselined_findings[0])

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        f'{{"version": 1, "entries": [{{"fingerprint": "{fp}", "owner": "alice", '
        f'"reason": "doc example", "expires": "2000-01-01"}}]}}'
    )
    baseline = load_baseline(str(baseline_path))
    assert baseline == {}  # expired entry is dropped entirely

    findings, baselined = scan_security(diff, baseline=baseline)
    assert len(findings) == 1
    assert baselined == []


def test_baseline_entry_without_expiry_is_honored():
    diff = make_diff("README.md", ["result = eval(user_input)"])
    unbaselined_findings = scan(diff)
    fp = compute_fingerprint(unbaselined_findings[0])
    baseline = {fp: {"owner": "alice", "reason": "doc example"}}  # no "expires" key

    findings, baselined = scan_security(diff, baseline=baseline)
    assert findings == []
    assert len(baselined) == 1


def test_baseline_excludes_matching_sensitive_file_finding():
    """Regression: sensitive-file findings are appended earlier in the scan
    loop than the rest and must still be checked against the baseline, not
    bypass it."""
    diff = make_diff("auth.py", ["x = 1"])
    unbaselined_findings = scan(diff)
    sf = [f for f in unbaselined_findings if f["category"] == "sensitive-file"][0]
    fp = compute_fingerprint(sf)
    baseline = {fp: {"owner": "alice", "reason": "known auth module", "expires": "2099-01-01"}}

    findings, baselined = scan_security(diff, baseline=baseline)
    assert not any(f["category"] == "sensitive-file" for f in findings)
    assert any(f["category"] == "sensitive-file" for f in baselined)


def test_baseline_fingerprint_is_specific_to_evidence_not_whole_file():
    """A baseline entry for one specific finding must not blanket-exempt a
    DIFFERENT finding in the same file."""
    diff = make_diff("README.md", ["result = eval(user_input)"])
    findings_no_baseline = scan(diff)
    wrong_fp = "0000000000000000"  # does not match the real finding
    baseline = {wrong_fp: {"owner": "alice", "reason": "unrelated"}}

    findings, baselined = scan_security(diff, baseline=baseline)
    assert len(findings) == len(findings_no_baseline)
    assert baselined == []


# --- redact_diff_for_llm: the diff sent to the external LLM must never
# contain a raw secret value, even though the human-facing report already
# redacts separately (this is a distinct leak path) ---

def test_redact_diff_for_llm_blanks_secret_line():
    diff = make_diff("config.py", ['API_KEY = "sk-live-realvalue123"'])
    findings = scan(diff)
    safe_diff = redact_diff_for_llm(diff, findings)
    assert "sk-live-realvalue123" not in safe_diff
    assert "REDACTED" in safe_diff


def test_redact_diff_for_llm_leaves_non_secret_lines_untouched():
    diff = make_diff("config.py", ["total = a + b"])
    findings = scan(diff)
    safe_diff = redact_diff_for_llm(diff, findings)
    assert safe_diff == diff


def test_redact_diff_for_llm_preserves_other_lines_on_a_mixed_diff():
    diff = make_diff("config.py", ['API_KEY = "sk-live-realvalue123"', "total = a + b"])
    findings = scan(diff)
    safe_diff = redact_diff_for_llm(diff, findings)
    assert "sk-live-realvalue123" not in safe_diff
    assert "total = a + b" in safe_diff
