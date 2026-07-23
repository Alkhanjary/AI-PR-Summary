import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT = str(Path(__file__).resolve().parent.parent / "summarize_pr.py")


def run_cli(args, diff_path, cwd=None):
    return subprocess.run(
        [sys.executable, SCRIPT, "--file", str(diff_path)] + args,
        capture_output=True, text=True, cwd=cwd,
    )


def write_diff(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def file_section(filename, added_lines):
    added = "".join(f"+{line}\n" for line in added_lines)
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"index 1111111..2222222 100644\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -1,1 +1,{1 + len(added_lines)} @@\n"
        f" context\n"
        f"{added}"
    )


VULNERABLE = file_section("runner.py", ["result = eval(user_input)"])
CLEAN = file_section("math_utils.py", ["total = a + b"])


def test_fail_on_high_exits_3_with_high_finding(tmp_path):
    diff = write_diff(tmp_path, "bad.diff", VULNERABLE)
    result = run_cli(["--security", "--dry-run", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_fail_on_high_exits_0_when_clean(tmp_path):
    diff = write_diff(tmp_path, "clean.diff", CLEAN)
    result = run_cli(["--security", "--dry-run", "--fail-on", "high"], diff)
    assert result.returncode == 0


def test_no_fail_on_reports_but_exits_0(tmp_path):
    diff = write_diff(tmp_path, "bad.diff", VULNERABLE)
    result = run_cli(["--security", "--dry-run"], diff)
    assert result.returncode == 0
    assert "dangerous-call" in result.stdout


def test_json_output_is_machine_readable(tmp_path):
    diff = write_diff(tmp_path, "bad.diff", VULNERABLE)
    result = run_cli(["--json"], diff)
    data = json.loads(result.stdout)
    assert data["version"] == 1
    assert data["counts"]["high"] >= 1
    assert any(f["category"] == "dangerous-call" for f in data["findings"])


def test_scan_covers_files_beyond_truncation_budget(tmp_path):
    # ~9 KB of harmless changes first, dangerous file last: smart_truncate
    # would omit auth.py from the LLM prompt, but the scan must still see it.
    filler = file_section("big_module.py", [f"safe_line_{i} = {i}" for i in range(400)])
    dangerous = file_section("auth.py", ["result = eval(user_input)"])
    assert len(filler) > 8000
    diff = write_diff(tmp_path, "big.diff", filler + dangerous)

    result = run_cli(["--security", "--dry-run", "--fail-on", "high"], diff)
    assert result.returncode == 3
    assert "auth.py" in result.stdout
    assert "dangerous-call" in result.stdout


def test_scan_covers_ignored_lockfiles(tmp_path):
    # filter_diff drops lockfiles from the LLM prompt; the scan must not.
    lockfile = file_section("package-lock.json", ['"npm_token": "a8f3k2j9d0s1x7c4"'])
    diff = write_diff(tmp_path, "lock.diff", lockfile)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3
    data = json.loads(result.stdout)
    assert any(f["file"] == "package-lock.json" for f in data["findings"])


def test_attacker_cannot_bypass_gate_by_naming_the_file_tests_or_docs(tmp_path):
    payload = file_section("tests/test_deploy.py", ['API_KEY = "sk-live-realvalue123"'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3
    data = json.loads(result.stdout)
    assert any(f["file"] == "tests/test_deploy.py" for f in data["findings"])


def test_nosec_cannot_bypass_gate_on_a_secret(tmp_path):
    # "# nosec" is attacker-controlled diff content — it must not be able to
    # wave away a finding, at any severity, just by being present.
    payload = file_section("config.py", ['API_KEY = "sk-live-realvalue123"  # nosec'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_nosec_cannot_bypass_medium_gate(tmp_path):
    # Reviewer-reported bypass: verify=False # nosec previously passed
    # --fail-on medium because nosec silently excluded it from the gate.
    payload = file_section("client.py", ["requests.get(url, verify=False)  # nosec"])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "medium"], diff)
    assert result.returncode == 3


def test_placeholder_variable_name_cannot_bypass_gate(tmp_path):
    payload = file_section("config.py", ['EXAMPLE_API_KEY = "sk-live-realvalue123"'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_placeholder_word_in_comment_cannot_bypass_gate(tmp_path):
    payload = file_section("config.py", ["TOKEN=abc123realtoken456 # example"])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_placeholder_substring_in_real_value_cannot_bypass_gate(tmp_path):
    payload = file_section("config.py", ['API_KEY = "sk-live-example-real123"'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_lowercase_unquoted_assignment_cannot_bypass_gate(tmp_path):
    payload = file_section("config.py", ["token=abc123realsecret456"])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_double_plus_prefix_cannot_bypass_gate(tmp_path):
    # diff marker "+" + content "++eval(user_input)" = raw line "+++eval(...)"
    body = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,2 @@\n"
        " context\n"
        "+++eval(user_input)\n"
    )
    diff = write_diff(tmp_path, "bad.diff", body)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 3


def test_baseline_file_excludes_matching_finding_from_gate(tmp_path):
    diff = write_diff(tmp_path, "bad.diff", VULNERABLE)

    # First run without a baseline to get the real fingerprint.
    result = run_cli(["--json"], diff)
    data = json.loads(result.stdout)
    from summarize_pr import compute_fingerprint  # local import: test-only convenience
    fp = compute_fingerprint(data["findings"][0])

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "version": 1,
        "entries": [{"fingerprint": fp, "owner": "alice", "reason": "test", "expires": "2099-01-01"}],
    }))

    result2 = run_cli(["--json", "--fail-on", "high", "--baseline-file", str(baseline_path)], diff)
    assert result2.returncode == 0
    data2 = json.loads(result2.stdout)
    assert data2["findings"] == []
    assert len(data2["baselined"]) == 1


def test_gitattributes_binary_marker_cannot_blind_read_input(tmp_path, monkeypatch):
    # Reviewer-reported bypass: a repo-level ".gitattributes" with "*.py -diff"
    # makes plain `git diff` print "Binary files ... differ" instead of the
    # real change, hiding it from the scanner entirely when the tool runs its
    # own `git diff` (no --file). --text/--no-ext-diff/--no-textconv must
    # force a real textual diff regardless.
    #
    # This exercises read_input() directly (in-process) rather than via a
    # subprocess CLI call: read_input() checks sys.stdin.isatty() before it
    # will use --base at all, and a subprocess's captured stdin is never a
    # TTY, so the --base code path can't be reached through the CLI in a test.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / ".gitattributes").write_text("*.py -diff\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)

    (repo / "app.py").write_text("def hello():\n    pass\n")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add app"], cwd=repo, check=True)

    (repo / "app.py").write_text("def hello():\n    pass\n\nresult = eval(user_input)\n")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add eval"], cwd=repo, check=True)

    from summarize_pr import read_input, scan_security

    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    diff_text = read_input(None, "HEAD~1")

    assert "Binary files" not in diff_text
    assert "eval(user_input)" in diff_text
    findings, _ = scan_security(diff_text)
    assert any(f["category"] == "dangerous-call" for f in findings)
