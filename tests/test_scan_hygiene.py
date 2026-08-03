"""Regression tests for issues found by running the vuln scanner on this repo.

Each of these was a real defect in how the dashboard handled its own scan
output, and each is easy to reintroduce, so they're pinned here rather than
left to the next manual scan to rediscover.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LLM_API_KEY", "test-key-not-used")

import server


# --- Findings inside the tool's own artifacts ------------------------------
# scan_history.json stores each past finding's `evidence` verbatim, so scanning
# a folder containing one re-reports every risky-looking line the scanner has
# ever seen, attributed to the JSON file. A 27-file scan produced six such
# findings, and the AI narrative built its remediation plan around them.

def test_self_artifact_excludes_saved_scan_history():
    assert server._is_self_artifact("scan_history.json")
    assert server._is_self_artifact(r"C:\projects\thing\scan_history.json")
    assert server._is_self_artifact("nested/dir/scan_history.json")


def test_self_artifact_excludes_editor_and_agent_config():
    assert server._is_self_artifact(r".claude\settings.local.json")
    assert server._is_self_artifact(".claude/settings.json")
    assert server._is_self_artifact(".vscode/launch.json")
    assert server._is_self_artifact("node_modules/pkg/index.js")


def test_self_artifact_keeps_real_source_files():
    for path in ("server.py", "dashboard/index.html", "tests/test_security.py",
                 ".env", "summarize_pr.py"):
        assert not server._is_self_artifact(path), path


def test_self_artifact_handles_missing_path():
    assert not server._is_self_artifact(None)
    assert not server._is_self_artifact("")


# --- Dismissed findings ----------------------------------------------------
# The AI review already judges some detections false positives. Leaving them
# mixed in meant the report's first three "critical" entries were the same AWS
# key in a redaction test.

def test_dismissed_detects_false_positive_verdicts():
    for verdict in ("false_positive", "FALSE-POSITIVE", "dismissed", "not_a_finding"):
        assert server._is_dismissed({"ai_verdict": verdict}), verdict


def test_dismissed_detects_test_fixtures():
    assert server._is_dismissed({"likely_test_fixture": True})


def test_dismissed_is_conservative():
    """Only an explicit dismissal counts - unreviewed and AI-confirmed findings
    must stay in the main report."""
    assert not server._is_dismissed({})
    assert not server._is_dismissed({"ai_verdict": "confirmed"})
    assert not server._is_dismissed({"ai_verdict": ""})
    assert not server._is_dismissed({"likely_test_fixture": False})


def test_report_splits_dismissed_into_appendix():
    real = {"file": "server.py", "line": 43, "severity": "critical",
            "category": "rce", "description": "eval on user input"}
    fixture = {"file": "tests/test_security.py", "line": 184, "severity": "critical",
               "category": "hardcoded-secret", "description": "AWS key",
               "ai_verdict": "false_positive", "ai_reason": "redaction test"}
    md = server.build_findings_markdown(
        {"findings": [real, fixture], "files_scanned": 27, "exit_code": 2}, {}, "")

    assert "Appendix — Possible False Positives" in md
    # Counts and the "start here" pointer describe real work, not fixtures.
    assert "| **Findings needing review** | **1** |" in md
    assert "Dismissed as likely false positives | 1" in md
    assert "server.py:43" in md.split("Appendix")[0]
    assert "test_security.py" not in md.split("Appendix")[0]
    # The dismissal reason is carried through so the call can be second-guessed.
    assert "redaction test" in md.split("Appendix")[1]


def test_report_omits_appendix_when_nothing_dismissed():
    real = {"file": "server.py", "line": 43, "severity": "high",
            "category": "xss", "description": "innerHTML"}
    md = server.build_findings_markdown({"findings": [real], "files_scanned": 1}, {}, "")
    assert "Appendix" not in md


# --- SSRF guard on the web scan's fetcher ----------------------------------
# probe_url fetches whatever URL the scan was pointed at. Scanning loopback and
# private ranges is the tool's actual purpose, so those stay allowed; cloud
# metadata endpoints and non-HTTP schemes never are.

def test_probe_allows_normal_scan_targets():
    for url in ("http://127.0.0.1:5000/", "http://localhost:8000/api",
                "https://example.com", "http://192.168.1.10/"):
        ok, why = server._probe_target_allowed(url)
        assert ok, f"{url} should be scannable ({why})"


def test_probe_blocks_cloud_metadata_endpoints():
    for url in ("http://169.254.169.254/latest/meta-data/",
                "http://metadata.google.internal/computeMetadata/v1/",
                "http://100.100.100.100/"):
        ok, _ = server._probe_target_allowed(url)
        assert not ok, url


def test_probe_blocks_non_http_schemes():
    for url in ("file:///c:/Windows/win.ini", "gopher://127.0.0.1:11211/_stats",
                "ftp://example.com/x", "not a url", "http://"):
        ok, _ = server._probe_target_allowed(url)
        assert not ok, url


def test_probe_url_refuses_without_making_a_request(monkeypatch):
    """The guard must run before requests.get, not after."""
    called = []
    monkeypatch.setattr(server.requests, "get",
                        lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError))
    out = server.probe_url("http://169.254.169.254/latest/meta-data/")
    assert called == []
    assert "refused to fetch" in out["error"]


def test_probe_url_revalidates_redirect_targets(monkeypatch):
    """A 302 into a blocked host must not be followed - the original bug was
    allow_redirects=True, which bypassed the check entirely."""
    class Resp:
        is_redirect = True
        is_permanent_redirect = False
        headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

    monkeypatch.setattr(server.requests, "get", lambda *a, **k: Resp())
    out = server.probe_url("http://127.0.0.1:5000/")
    assert "refused to follow redirect" in out["error"]


def test_probe_url_stops_on_redirect_loop(monkeypatch):
    class Resp:
        is_redirect = True
        is_permanent_redirect = False
        headers = {"Location": "http://127.0.0.1:5000/next"}

    monkeypatch.setattr(server.requests, "get", lambda *a, **k: Resp())
    out = server.probe_url("http://127.0.0.1:5000/")
    assert "redirects" in out["error"]


# --- Saved history lives outside any scanned tree --------------------------

def test_scan_history_is_stored_outside_the_repo():
    repo_root = Path(server.__file__).resolve().parent
    assert repo_root not in server.SCAN_HISTORY_FILE.resolve().parents, (
        "scan_history.json must not sit inside the repo, or scanning this "
        "project re-detects every finding it has ever stored"
    )
