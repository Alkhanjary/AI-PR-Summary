import json
import subprocess
import sys
from pathlib import Path

SCRIPT = str(Path(__file__).resolve().parent.parent / "summarize_pr.py")


def run_cli(args, diff_path):
    return subprocess.run(
        [sys.executable, SCRIPT, "--file", str(diff_path)] + args,
        capture_output=True, text=True,
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


def test_fail_on_high_exits_2_with_high_finding(tmp_path):
    diff = write_diff(tmp_path, "bad.diff", VULNERABLE)
    result = run_cli(["--security", "--dry-run", "--fail-on", "high"], diff)
    assert result.returncode == 2


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
    assert result.returncode == 2
    assert "auth.py" in result.stdout
    assert "dangerous-call" in result.stdout


def test_scan_covers_ignored_lockfiles(tmp_path):
    # filter_diff drops lockfiles from the LLM prompt; the scan must not.
    lockfile = file_section("package-lock.json", ['"npm_token": "a8f3k2j9d0s1x7c4"'])
    diff = write_diff(tmp_path, "lock.diff", lockfile)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert any(f["file"] == "package-lock.json" for f in data["findings"])


def test_attacker_cannot_bypass_gate_by_naming_the_file_tests_or_docs(tmp_path):
    # Reviewer-reported bypass: a real credential added to tests/test_deploy.py
    # produced zero findings because tests/docs/*.md were excluded wholesale.
    payload = file_section("tests/test_deploy.py", ['API_KEY = "sk-live-realvalue123"'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert any(f["file"] == "tests/test_deploy.py" for f in data["findings"])


def test_attacker_cannot_bypass_gate_with_inline_nosec_on_a_secret(tmp_path):
    # "# nosec" is attacker-controlled diff content — it must not be able to
    # wave away a high-confidence finding like a hardcoded secret.
    payload = file_section("config.py", ['API_KEY = "sk-live-realvalue123"  # nosec'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 2


def test_attacker_cannot_bypass_gate_with_placeholder_variable_name(tmp_path):
    # Reviewer-reported bypass: EXAMPLE_API_KEY with a real value produced
    # zero findings because placeholder detection checked the whole match
    # (including the variable name), not just the captured value.
    payload = file_section("config.py", ['EXAMPLE_API_KEY = "sk-live-realvalue123"'])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 2


def test_attacker_cannot_bypass_gate_with_placeholder_word_in_comment(tmp_path):
    # Reviewer-reported bypass: TOKEN=... # example produced zero findings
    # because the unquoted-credential regex's whole match (including the
    # trailing comment) was checked for placeholder words.
    payload = file_section("config.py", ["TOKEN=abc123realtoken456 # example"])
    diff = write_diff(tmp_path, "bad.diff", payload)

    result = run_cli(["--json", "--fail-on", "high"], diff)
    assert result.returncode == 2
