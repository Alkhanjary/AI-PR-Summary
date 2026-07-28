import io
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_pr import (
    filter_diff,
    smart_truncate,
    scan_security,
    format_security_report,
    call_llm,
    SYSTEM_PROMPT,
    VULN_REPORT_SYSTEM_PROMPT,
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


@app.route("/api/github/user")
def github_user():
    """Powers the Overview tab's user profile card (name, bio, follower
    counts, account age, etc.) - separate from /api/github/repos, which
    only returns the repo list used to compute aggregate stats."""
    username = request.args.get("username", "")
    if not username:
        return jsonify({"error": "username required"}), 400
    r = requests.get(
        "https://api.github.com/users/" + username,
        headers=gh_headers(),
    )
    return jsonify(r.json()), r.status_code


@app.route("/api/github/user-activity")
def github_user_activity():
    """Powers the Overview tab's 'Involved in' section: pull requests the
    user has authored across ANY repo (not just ones they own), so
    contributions to other people's/orgs' repos show up too."""
    username = request.args.get("username", "")
    if not username:
        return jsonify({"error": "username required"}), 400
    r = requests.get(
        "https://api.github.com/search/issues",
        params={"q": "author:" + username + " is:pr", "sort": "updated", "order": "desc", "per_page": 20},
        headers=gh_headers(),
    )
    if not r.ok:
        return jsonify({"items": []}), r.status_code
    data = r.json()
    items = []
    for item in data.get("items", []):
        repo_url = item.get("repository_url", "")
        repo_full_name = "/".join(repo_url.split("/")[-2:]) if repo_url else ""
        items.append({
            "repo": repo_full_name,
            "title": item.get("title"),
            "html_url": item.get("html_url"),
            "state": "merged" if item.get("pull_request", {}).get("merged_at") else item.get("state"),
            "updated_at": item.get("updated_at"),
        })
    return jsonify({"items": items})


@app.route("/api/github/search-users")
def github_search_users():
    """Powers the username autocomplete dropdown: proxies GitHub's user
    search so the frontend doesn't need its own token/CORS handling."""
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify({"items": []})
    r = requests.get(
        "https://api.github.com/search/users",
        params={"q": q, "per_page": 6},
        headers=gh_headers(),
    )
    if not r.ok:
        return jsonify({"items": []}), r.status_code
    data = r.json()
    items = [
        {"login": item["login"], "avatar_url": item.get("avatar_url", "")}
        for item in data.get("items", [])
    ]
    return jsonify({"items": items})


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


@app.route("/api/team-stats")
def team_stats():
    """Aggregates commit activity per contributor (commits, lines added,
    lines deleted) for the given repo, using GitHub's API for the commit
    list and per-commit stats. Powers the Monitor tab's team overview.
    Optional ?since=<ISO8601> filters to commits after that date, so the
    dashboard can offer day/week/month/all-time views."""
    repo = request.args.get("repo", "").strip()
    limit = int(request.args.get("limit", 100))
    since = request.args.get("since", "").strip()
    if not repo:
        return jsonify({"error": "repo required"}), 400

    until = request.args.get("until", "").strip()
    params = {"per_page": min(limit, 100)}
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    r = requests.get(
        "https://api.github.com/repos/" + repo + "/commits",
        params=params,
        headers=gh_headers(),
    )
    if r.status_code != 200:
        return jsonify(r.json()), r.status_code
    commits = r.json()

    stats = {}
    timeline = []
    for c in commits:
        author = c.get("author")
        login = author["login"] if author else (c["commit"]["author"]["name"] if c["commit"].get("author") else "unknown")
        avatar = author["avatar_url"] if author else None
        date = c["commit"]["author"]["date"] if c["commit"].get("author") else None
        if login not in stats:
            stats[login] = {"login": login, "avatar_url": avatar, "commits": 0, "additions": 0, "deletions": 0, "test_commits": 0, "active_days": set()}
        stats[login]["commits"] += 1
        if date:
            stats[login]["active_days"].add(date[:10])

        detail = requests.get(
            "https://api.github.com/repos/" + repo + "/commits/" + c["sha"],
            headers=gh_headers(),
        )
        commit_additions = commit_deletions = 0
        if detail.status_code == 200:
            d = detail.json()
            s = d.get("stats", {})
            commit_additions = s.get("additions", 0)
            commit_deletions = s.get("deletions", 0)
            stats[login]["additions"] += commit_additions
            stats[login]["deletions"] += commit_deletions
            touched_files = [f.get("filename", "") for f in d.get("files", [])]
            if any("test" in fn.lower() for fn in touched_files):
                stats[login]["test_commits"] += 1

        if date:
            timeline.append({"login": login, "date": date[:10], "additions": commit_additions, "deletions": commit_deletions})

    for v in stats.values():
        v["avg_lines_per_commit"] = round((v["additions"] + v["deletions"]) / v["commits"], 1) if v["commits"] else 0
        v["test_coverage_pct"] = round(100 * v["test_commits"] / v["commits"], 1) if v["commits"] else 0
        v["active_days_count"] = len(v["active_days"])
        del v["active_days"]

    return jsonify({"contributors": sorted(stats.values(), key=lambda x: -x["commits"]), "timeline": timeline})


TEAM_BUG_JOBS = {}


def run_team_bugs_job(job_id, repo, use_ai):
    job = TEAM_BUG_JOBS[job_id]
    try:
        if not ensure_scanner_available():
            job["status"] = "error"
            job["error"] = "Could not find or clone the scanner tool"
            return

        target_dir = get_repo_dir(repo)
        job["log"].append(f"Target ready: {target_dir}")

        import tempfile
        tmpdir = tempfile.mkdtemp()
        json_out = str(Path(tmpdir) / "report.json")
        cmd = ["py", str(OTHER_SCANNER_PATH), str(target_dir), "--json", json_out, "--scan", "code", "--fail-on", "none"]
        if use_ai:
            cmd.append("--ai")

        scan_env = os.environ.copy()
        scan_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, cwd=OTHER_SCANNER_PATH.parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env=scan_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job["log"].append(line)
        proc.wait()

        try:
            with open(json_out, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            job["status"] = "error"
            job["error"] = "Scan did not produce a valid report."
            return

        job["log"].append("Attributing findings via git blame...")
        severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        contributors = {}
        for finding in report.get("findings", []):
            file_path = finding.get("file", "")
            fline = finding.get("line")
            if not file_path or not fline:
                continue
            if file_path.lower().endswith((".md", ".rst", ".txt")):
                continue
            if use_ai and finding.get("ai_verdict") == "false_positive":
                continue

            blame = subprocess.run(
                ["git", "blame", "-L", f"{fline},{fline}", "--porcelain", "--", file_path],
                cwd=target_dir, capture_output=True, text=True,
            )
            author = "unknown"
            for bline in blame.stdout.splitlines():
                if bline.startswith("author "):
                    author = bline[len("author "):].strip()
                    break

            if author not in contributors:
                contributors[author] = {"author": author, "total_bugs": 0, "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0}, "worst": "low", "findings": []}
            c = contributors[author]
            c["total_bugs"] += 1
            sev = finding.get("severity", "low")
            if sev in c["by_severity"]:
                c["by_severity"][sev] += 1
            if severity_rank.get(sev, 0) > severity_rank.get(c["worst"], 0):
                c["worst"] = sev
            c["findings"].append({
                "file": file_path,
                "line": fline,
                "severity": sev,
                "description": finding.get("description", ""),
                "evidence": finding.get("evidence", ""),
                "impact": finding.get("impact", ""),
                "improvement": finding.get("improvement", ""),
            })

        job["status"] = "done"
        job["result"] = sorted(contributors.values(), key=lambda x: -x["total_bugs"])
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/api/team-summary", methods=["POST"])
def team_summary():
    """Generates a short, plain-English executive summary from
    aggregated team stats, using the same LLM setup as the rest of
    this tool. Purely descriptive; never used for any gating."""
    data = request.get_json(force=True)
    client, model = get_client()
    if not client:
        return jsonify({"error": "LLM_API_KEY not configured on the server."}), 400

    prompt = (
        "You are summarizing a software team's activity for a manager/executive. "
        "Write 2-3 short sentences, plain English, no markdown headers, no bullet points. "
        "Be factual and neutral, note both progress and any risk. Here is the data:\n\n"
        + json.dumps(data, indent=2)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write brief, factual team-activity summaries for a manager. 2-3 sentences max, no formatting."},
                {"role": "user", "content": prompt},
            ],
        )
        return jsonify({"summary": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/team-bugs/start", methods=["POST"])
def team_bugs_start():
    """Starts the bug-attribution scan as a background job (mirrors
    /api/vuln-scan/start), so the Monitor tab can show live progress
    instead of blocking, especially when AI verification is enabled."""
    data = request.get_json(force=True)
    repo = data.get("repo", "").strip()
    use_ai = data.get("ai", False)
    if not repo:
        return jsonify({"error": "repo required"}), 400

    job_id = str(uuid.uuid4())
    TEAM_BUG_JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None, "started": time.time()}

    thread = threading.Thread(target=run_team_bugs_job, args=(job_id, repo, use_ai), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/team-bugs/status/<job_id>")
def team_bugs_status(job_id):
    job = TEAM_BUG_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify({
        "status": job["status"],
        "log": job["log"],
        "result": job["result"],
        "error": job["error"],
        "elapsed": round(time.time() - job["started"], 1),
    })


@app.route("/api/branches")
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


OTHER_SCANNER_REPO = "https://github.com/Alkhanjary/security-scan.git"
OTHER_SCANNER_PATH = Path(__file__).resolve().parent.parent / "security-scan" / "scanner.py"


def ensure_scanner_available():
    """If the security-scan tool isn't present locally, clone it fresh
    from GitHub. Keeps the two tools in separate folders (never merged),
    but self-heals if the local copy is missing (new machine, deleted
    folder, etc.) instead of just failing."""
    if OTHER_SCANNER_PATH.exists():
        return True
    scanner_dir = OTHER_SCANNER_PATH.parent
    if not scanner_dir.exists():
        subprocess.run(
            ["git", "clone", OTHER_SCANNER_REPO, str(scanner_dir)],
            capture_output=True, text=True,
        )
    return OTHER_SCANNER_PATH.exists()


import threading
import time
import uuid

VULN_JOBS = {}


def run_vuln_scan_job(job_id, repo, use_ai, scan_types, fail_on, include_test_files, files=None,
                       url=None, net_target=None, net_ports=None):
    job = VULN_JOBS[job_id]
    try:
        import tempfile
        if files:
            upload_dir = Path(tempfile.mkdtemp())
            for f in files:
                rel_path = f.get("path", "").lstrip("/\\")
                if not rel_path:
                    continue
                dest = upload_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(f.get("content", ""), encoding="utf-8", errors="replace")
            target_dir = upload_dir
            job["log"].append(f"Uploaded {len(files)} file(s) to scan: {target_dir}")
        else:
            target_dir = get_repo_dir(repo)
            job["log"].append(f"Target ready: {target_dir}")

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
        if url:
            cmd += ["--url", url]
        if net_target:
            cmd += ["--net-target", net_target]
        if net_ports:
            cmd += ["--net-ports", net_ports]

        job["log"].append("Starting scan: " + " ".join(cmd))

        scan_env = os.environ.copy()
        scan_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, cwd=OTHER_SCANNER_PATH.parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env=scan_env,
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
    files = data.get("files")
    if not ensure_scanner_available():
        return jsonify({"error": f"Could not find or clone the scanner tool (expected at {OTHER_SCANNER_PATH})"}), 500
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
            files,
        ),
        kwargs={
            "url": (data.get("url") or "").strip() or None,
            "net_target": (data.get("net_target") or "").strip() or None,
            "net_ports": (data.get("net_ports") or "").strip() or None,
        },
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


REPORT_MAX_CHARS = 60000

SEVERITY_ORDER = ["critical", "high", "medium", "low"]
SEVERITY_COLORS_HEX = {"critical": "4c0519", "high": "dc2626", "medium": "b45309", "low": "15803d"}


def count_findings_by_severity(findings):
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        sev = (f.get("severity") or "").lower()
        if sev in counts:
            counts[sev] += 1
    return counts


def build_findings_markdown(result):
    """Deterministically builds the Findings Summary + Detailed Findings
    sections directly from the scanner's JSON, so every single finding is
    guaranteed to appear in the report - not left to the LLM to enumerate,
    which risks silently dropping findings from a long list. Findings are
    grouped by severity (critical first) with a clear subsection per
    severity, so the document reads as an organized triage list rather
    than a flat dump."""
    findings = result.get("findings", [])
    counts = count_findings_by_severity(findings)

    category_counts = {}
    for f in findings:
        cat = f.get("category") or f.get("rule") or "uncategorized"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    lines = ["## Findings Summary", ""]
    lines.append(f"- **Files scanned:** {result.get('files_scanned', 'n/a')}")
    lines.append(f"- **Total findings:** {len(findings)}")
    for sev in SEVERITY_ORDER:
        lines.append(f"- **{sev.capitalize()}:** {counts[sev]}")
    if category_counts:
        lines.append("")
        lines.append("**By category:**")
        for cat, n in sorted(category_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {cat}: {n}")

    skipped = result.get("skipped_files") or []
    ai_errors = result.get("ai_scan_errors") or []
    if result.get("truncated") or skipped or ai_errors:
        lines.append("")
        lines.append("**Scan notes:**")
        if result.get("truncated"):
            lines.append("- Scan output was truncated; some files may not have been fully analyzed.")
        if skipped:
            lines.append(f"- {len(skipped)} file(s) were skipped during the scan (unsupported type, too large, or unreadable).")
        if ai_errors:
            lines.append(f"- AI verification failed for {len(ai_errors)} file(s); their findings are unverified.")
    lines.append("")

    def sort_key(f):
        sev = (f.get("severity") or "low").lower()
        return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else len(SEVERITY_ORDER)

    grouped = {sev: [] for sev in SEVERITY_ORDER}
    other = []
    for f in findings:
        sev = (f.get("severity") or "").lower()
        (grouped[sev] if sev in grouped else other).append(f)

    lines.append("## Detailed Findings")
    lines.append("")

    for sev in SEVERITY_ORDER + (["other"] if other else []):
        group = other if sev == "other" else grouped[sev]
        if not group:
            continue
        label = "Other" if sev == "other" else sev.capitalize()
        lines.append(f"### {label} severity ({len(group)})")
        lines.append("")
        for f in group:
            sev_tag = (f.get("severity") or "unknown").upper()
            file_ = f.get("file", "unknown file")
            line_no = f.get("line", "?")
            category = f.get("category") or f.get("rule")
            heading = f"#### [{sev_tag}] {file_}:{line_no}"
            if category:
                heading += f" &mdash; {category}"
            lines.append(heading)
            lines.append("")
            lines.append(f"- **Location:** `{file_}`, line {line_no}")
            if f.get("description"):
                lines.append(f"- **Issue:** {f['description']}")
            if f.get("evidence"):
                lines.append(f"- **Evidence:** `{f['evidence']}`")
            if f.get("impact"):
                lines.append(f"- **Impact:** {f['impact']}")
            if f.get("improvement"):
                lines.append(f"- **Fix:** {f['improvement']}")
            if f.get("ai_verdict"):
                verdict_line = f"- **AI verdict:** {f['ai_verdict']}"
                if f.get("ai_reason"):
                    verdict_line += f" ({f['ai_reason']})"
                lines.append(verdict_line)
            if f.get("likely_test_fixture"):
                lines.append("- _Likely a test fixture, not production code._")
            lines.append("")

    return "\n".join(lines)


def parse_markdown_blocks(md_text):
    """Small markdown parser tuned to the structure this report's sections
    produce (## / ### headings, "- " bullets, "1. " numbered lists,
    paragraphs). Returns a list of (block_type, text) tuples for the
    docx/pdf builders."""
    blocks = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("#### "):
            blocks.append(("h4", line[5:].strip()))
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
        elif line.lstrip().startswith("- "):
            blocks.append(("bullet", line.lstrip()[2:].strip()))
        elif re.match(r"^\d+\.\s", line.lstrip()):
            blocks.append(("numbered", re.sub(r"^\d+\.\s", "", line.lstrip())))
        else:
            blocks.append(("p", line.strip()))
    return blocks


def strip_md_inline(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


FINDING_HEADING_RE = re.compile(r"^\[(CRITICAL|HIGH|MEDIUM|LOW|UNKNOWN)\]\s*(.*)$")


def build_report_docx(md_text, title, subtitle=None):
    from docx import Document
    from docx.shared import RGBColor

    doc = Document()
    doc.add_heading(title, level=0)
    if subtitle:
        sub = doc.add_paragraph(subtitle)
        sub.runs[0].italic = True

    for kind, text in parse_markdown_blocks(md_text):
        if kind == "h4":
            m = FINDING_HEADING_RE.match(text)
            p = doc.add_paragraph(style="Heading 4")
            if m:
                sev, rest = m.group(1), strip_md_inline(m.group(2))
                hexcolor = SEVERITY_COLORS_HEX.get(sev.lower(), "374151")
                run = p.add_run(sev + "  ")
                run.font.color.rgb = RGBColor.from_string(hexcolor)
                p.add_run(rest)
            else:
                p.add_run(strip_md_inline(text))
            continue
        text = strip_md_inline(text)
        if kind == "h1":
            doc.add_heading(text, level=1)
        elif kind == "h2":
            doc.add_heading(text, level=2)
        elif kind == "h3":
            doc.add_heading(text, level=3)
        elif kind == "bullet":
            doc.add_paragraph(text, style="List Bullet")
        elif kind == "numbered":
            doc.add_paragraph(text, style="List Number")
        else:
            doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_report_pdf(md_text, title, subtitle=None):
    from xml.sax.saxutils import escape as xml_escape
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    heading3_finding_style = ParagraphStyle(
        "Heading3Finding", parent=styles["Heading3"], spaceBefore=14, spaceAfter=4, fontSize=12,
    )
    story = [Paragraph(xml_escape(title), styles["Title"])]
    if subtitle:
        story.append(Paragraph(xml_escape(subtitle), styles["Italic"]))
    story.append(Spacer(1, 12))

    for kind, text in parse_markdown_blocks(md_text):
        if kind == "h4":
            m = FINDING_HEADING_RE.match(text)
            if m:
                sev, rest = m.group(1), xml_escape(strip_md_inline(m.group(2)))
                hexcolor = SEVERITY_COLORS_HEX.get(sev.lower(), "374151")
                para_text = f'<font color="#{hexcolor}"><b>{xml_escape(sev)}</b></font>&nbsp;&nbsp;{rest}'
            else:
                para_text = xml_escape(strip_md_inline(text))
            story.append(Paragraph(para_text, heading3_finding_style))
            continue
        text = xml_escape(strip_md_inline(text))
        if kind == "h1":
            story.append(Spacer(1, 10))
            story.append(Paragraph(text, styles["Heading1"]))
        elif kind == "h2":
            story.append(Spacer(1, 8))
            story.append(Paragraph(text, styles["Heading2"]))
        elif kind == "h3":
            story.append(Spacer(1, 6))
            story.append(Paragraph(text, styles["Heading3"]))
        elif kind in ("bullet", "numbered"):
            story.append(Paragraph("• " + text, styles["Normal"]))
        else:
            story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()


def build_report_html(md_text, title, subtitle=None):
    import html as html_lib
    import markdown as md_lib

    def colorize_heading(m):
        sev = m.group(1)
        hexcolor = SEVERITY_COLORS_HEX.get(sev.lower(), "374151")
        return (
            f'#### <span style="background:#{hexcolor}22;color:#{hexcolor};'
            f'padding:2px 10px;border-radius:999px;font-size:0.68em;'
            f'font-weight:700;letter-spacing:0.03em;">{sev}</span>'
        )

    md_text = re.sub(r"^#### \[(CRITICAL|HIGH|MEDIUM|LOW|UNKNOWN)\]", colorize_heading, md_text, flags=re.MULTILINE)
    body = md_lib.markdown(md_text, extensions=["extra"])
    safe_title = html_lib.escape(title)
    # subtitle is built server-side with a literal `&middot;` entity, not
    # user input, so it's passed through as-is rather than re-escaped
    # (escaping would turn it into the literal text "&middot;").
    safe_subtitle = subtitle or ""
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{safe_title}</title><style>"
        "body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:860px;"
        "margin:2rem auto;padding:0 1.5rem;color:#1f2937;line-height:1.6;}"
        "h1{margin-bottom:2px;}"
        ".report-subtitle{color:#9ca3af;font-size:13px;margin-bottom:1.5rem;}"
        "h2{color:#111827;border-bottom:1px solid #e5e7eb;padding-bottom:4px;margin-top:2.2rem;}"
        "h3{margin-top:1.8rem;color:#374151;}"
        "h4{display:flex;align-items:center;gap:10px;margin-top:1.4rem;font-size:15px;}"
        "code{background:#f3f4f6;padding:1px 4px;border-radius:4px;}"
        "ul{padding-left:1.3rem;}"
        f"</style></head><body><h1>{safe_title}</h1>"
        f"<div class=\"report-subtitle\">{safe_subtitle}</div>{body}</body></html>"
    )


@app.route("/api/vuln-scan/report", methods=["POST"])
def vuln_scan_report():
    """Generates a detailed vulnerability report from a completed scan: an
    AI-written narrative (executive summary, risk overview, remediation
    plan) followed by a deterministically-built, complete list of every
    finding - so nothing gets summarized away. Returned as a downloadable
    file in the requested format."""
    data = request.get_json(force=True)
    job_id = data.get("job_id", "")
    fmt = (data.get("format") or "markdown").lower()

    job = VULN_JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("result"):
        return jsonify({"error": "No completed scan found for that job_id."}), 400

    result = job["result"]
    findings = result.get("findings", [])
    if not findings:
        return jsonify({"error": "No findings to report on."}), 400

    client, model = get_client()
    if not client:
        return jsonify({"error": "LLM_API_KEY is not configured on the server."}), 500

    counts = count_findings_by_severity(findings)
    summary_payload = {
        "files_scanned": result.get("files_scanned"),
        "exit_code": result.get("exit_code"),
        "counts": counts,
        "findings": findings,
    }
    findings_json = json.dumps(summary_payload, indent=2)[:REPORT_MAX_CHARS]
    try:
        narrative_md = call_llm(client, model, findings_json, system_prompt=VULN_REPORT_SYSTEM_PROMPT)
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {e}"}), 500

    report_md = narrative_md.strip() + "\n\n" + build_findings_markdown(result)

    title = "Vulnerability Assessment Report"
    subtitle = (
        f"{result.get('files_scanned', 'n/a')} files scanned &middot; {len(findings)} findings "
        f"&middot; generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    subtitle_plain = subtitle.replace("&middot;", "-")

    if fmt == "markdown":
        return Response(report_md, mimetype="text/markdown",
                         headers={"Content-Disposition": "attachment; filename=vuln-report.md"})
    if fmt == "html":
        return Response(build_report_html(report_md, title, subtitle), mimetype="text/html",
                         headers={"Content-Disposition": "attachment; filename=vuln-report.html"})
    if fmt == "docx":
        return Response(build_report_docx(report_md, title, subtitle_plain),
                         mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                         headers={"Content-Disposition": "attachment; filename=vuln-report.docx"})
    if fmt == "pdf":
        return Response(build_report_pdf(report_md, title, subtitle_plain), mimetype="application/pdf",
                         headers={"Content-Disposition": "attachment; filename=vuln-report.pdf"})

    return jsonify({"error": f"Unknown format: {fmt}"}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
