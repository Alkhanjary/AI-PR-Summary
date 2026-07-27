import json
import subprocess
import sys
from pathlib import Path

import requests

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_pr import (
    filter_diff,
    smart_truncate,
    scan_security,
    format_security_report,
    call_llm,
    SYSTEM_PROMPT,
    MAX_CHARS,
)
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
CORS(app)


def get_client():
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None
    if not api_key:
        return None, None
    model = os.environ.get("LLM_MODEL", "gpt-oss:20b-cloud")
    return OpenAI(api_key=api_key, base_url=base_url), model


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/github/repos")
def github_repos():
    username = request.args.get("username", "")
    if not username:
        return jsonify({"error": "username required"}), 400
    r = requests.get(
        "https://api.github.com/users/" + username + "/repos",
        params={"per_page": 100, "sort": "updated", "direction": "desc"},
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


CACHE_DIR = Path(__file__).resolve().parent / ".repo-cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_repo_dir(repo):
    """Returns a local working copy of the given owner/repo, cloning it
    (or fetching updates if already cloned) into a local cache folder.
    This is what lets the Run scan tab follow whichever repo is picked
    up top, not just the AI-PR-Summary folder this server lives in."""
    if not repo:
        return Path(__file__).resolve().parent
    local_path = CACHE_DIR / repo.replace("/", "__")
    if not local_path.exists():
        subprocess.run(
            ["git", "clone", "--no-single-branch", f"https://github.com/{repo}.git", str(local_path)],
            capture_output=True, text=True,
        )
    else:
        subprocess.run(["git", "fetch", "--all", "--quiet"], capture_output=True, text=True, cwd=local_path)
    return local_path


@app.route("/api/branches")
def branches():
    """Lists ALL remote branches for the given repo (cloning/fetching it
    into a local cache first), so the dashboard can offer a GitHub-style
    base/compare selector for whichever repo is currently selected."""
    repo = request.args.get("repo", "").strip()
    cwd = get_repo_dir(repo)
    result = subprocess.run(
        ["git", "branch", "-r", "--format=%(refname:short)"], capture_output=True, text=True, cwd=cwd
    )
    current = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=cwd)
    branch_list = [
        b.strip() for b in result.stdout.splitlines()
        if b.strip() and not b.strip().endswith("HEAD")
    ]
    return jsonify({"branches": branch_list, "current": current.stdout.strip()})


@app.route("/api/current-diff")
def current_diff():
    """Runs git diff for the given repo (cloning/fetching it into a
    local cache first) and returns it, so the dashboard can get a diff
    for whichever repo is selected, not just the local AI-PR-Summary
    checkout. With no ?base=, returns uncommitted changes only (only
    meaningful for the AI-PR-Summary repo this server lives in).
    With ?base=main&compare=my-branch, returns the full diff between
    the two named branches, matching what a real PR would contain."""
    repo = request.args.get("repo", "").strip()
    base = request.args.get("base", "").strip()
    compare = request.args.get("compare", "HEAD").strip() or "HEAD"
    cwd = get_repo_dir(repo)
    if base:
        result = subprocess.run(
            ["git", "diff", f"{base}...{compare}"], capture_output=True, text=True, cwd=cwd
        )
    else:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=cwd)
    return jsonify({"diff": result.stdout})


@app.route("/api/scan", methods=["POST"])
def scan():
    """Runs the real summarize_pr.py pipeline: filter -> truncate -> LLM."""
    data = request.get_json(force=True)
    diff_text = data.get("diff", "")

    if not diff_text.strip():
        return jsonify({"error": "Diff is empty."}), 400

    filtered = filter_diff(diff_text)
    if not filtered.strip():
        return jsonify({"error": "Diff only contained ignored files (lockfiles, minified assets)."}), 400

    truncated, omitted = smart_truncate(filtered, MAX_CHARS)

    client, model = get_client()
    if not client:
        return jsonify({"error": "LLM_API_KEY not configured on the server."}), 400

    try:
        summary = call_llm(client, model, truncated)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    return jsonify({"markdown": summary, "omitted_files": omitted})


@app.route("/api/security-scan", methods=["POST"])
def security_scan():
    """Runs the deterministic security scanner on the full raw diff."""
    data = request.get_json(force=True)
    diff_text = data.get("diff", "")

    if not diff_text.strip():
        return jsonify({"error": "Diff is empty."}), 400

    findings, baselined = scan_security(diff_text)
    report = format_security_report(findings, baselined)

    return jsonify({"markdown": report, "findings": findings, "baselined": baselined})


def gh_headers():
    token = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": "Bearer " + token} if token else {}


@app.route("/api/github/pulls")
def github_pulls():
    repo = request.args.get("repo", "")
    if "/" not in repo:
        return jsonify({"error": "repo must be owner/repo"}), 400
    r = requests.get(
        "https://api.github.com/repos/" + repo + "/pulls",
        params={"state": "all", "per_page": 10, "sort": "updated", "direction": "desc"},
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


@app.route("/api/github/comments")
def github_comments():
    repo = request.args.get("repo", "")
    number = request.args.get("number", "")
    r = requests.get(
        "https://api.github.com/repos/" + repo + "/issues/" + number + "/comments",
        params={"per_page": 20},
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


@app.route("/api/github/commits")
def github_commits():
    repo = request.args.get("repo", "")
    r = requests.get(
        "https://api.github.com/repos/" + repo + "/commits",
        params={"per_page": 10},
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


@app.route("/api/github/commit/<sha>")
def github_commit_detail(sha):
    repo = request.args.get("repo", "")
    r = requests.get(
        "https://api.github.com/repos/" + repo + "/commits/" + sha,
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


OTHER_SCANNER_PATH = Path(__file__).resolve().parent.parent / "security-scan" / "scanner.py"


import threading
import time
import uuid

VULN_JOBS = {}


def run_vuln_scan_job(job_id, repo, use_ai, scan_types, fail_on, include_test_files):
    job = VULN_JOBS[job_id]
    try:
        target_dir = get_repo_dir(repo)
        job["log"].append(f"Target ready: {target_dir}")

        import tempfile
        tmpdir = tempfile.mkdtemp()
        json_out = str(Path(tmpdir) / "report.json")
        scan_arg = ",".join(scan_types) if scan_types else "code"
        cmd = [
            "py", str(OTHER_SCANNER_PATH), str(target_dir),
            "--json", json_out,
            "--scan", scan_arg,
            "--fail-on", fail_on,
        ]
        if use_ai:
            cmd.append("--ai")
        if include_test_files:
            cmd.append("--include-test-files")

        job["log"].append("Starting scan: " + " ".join(cmd))

        proc = subprocess.Popen(
            cmd, cwd=OTHER_SCANNER_PATH.parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job["log"].append(line)
        proc.wait()

        try:
            with open(json_out, "r", encoding="utf-8") as f:
                report = json.load(f)
            job["status"] = "done"
            job["result"] = report
        except (FileNotFoundError, json.JSONDecodeError):
            job["status"] = "error"
            job["error"] = "Scanner did not produce a valid report."
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/api/vuln-scan/start", methods=["POST"])
def vuln_scan_start():
    """Starts the vulnerability scan as a background job and returns a
    job_id immediately, so the dashboard can poll for live progress
    instead of blocking on one long request (AI verification especially
    can take a while)."""
    data = request.get_json(force=True)
    repo = data.get("repo", "").strip()

    if not OTHER_SCANNER_PATH.exists():
        return jsonify({"error": f"Scanner not found at {OTHER_SCANNER_PATH}"}), 500

    job_id = str(uuid.uuid4())
    VULN_JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None, "started": time.time()}

    thread = threading.Thread(
        target=run_vuln_scan_job,
        args=(
            job_id, repo,
            data.get("ai", True),
            data.get("scan_types", ["code"]),
            data.get("fail_on", "high"),
            data.get("include_test_files", False),
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/vuln-scan/status/<job_id>")
def vuln_scan_status(job_id):
    job = VULN_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify({
        "status": job["status"],
        "log": job["log"],
        "result": job["result"],
        "error": job["error"],
        "elapsed": round(time.time() - job["started"], 1),
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
