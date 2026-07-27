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


@app.route("/api/current-diff")
def current_diff():
    """Runs git diff in this repo and returns it, so the dashboard can
    offer 'load my current uncommitted changes' without the browser
    needing filesystem/git access itself."""
    result = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
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


if __name__ == "__main__":
    app.run(port=5000, debug=True)
