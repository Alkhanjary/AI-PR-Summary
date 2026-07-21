import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import OpenAI

MAX_CHARS = 8000

IGNORED_PATTERNS = [
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    ".min.js",
    ".min.css",
]

SYSTEM_PROMPT = """You review code diffs for PR reviewers.

Return ONLY markdown with:

## Suggested Title
One concise conventional-commit style PR title (e.g. "fix: handle empty diff input")

## Summary
- 2 to 4 short bullets of what changed

## Impact
- 1 to 3 short bullets explaining what this change actually does, the resulting behavior or functional effect. Focus on outcome, not mechanics.

## Risk
low | medium | high (one word, then one short reason)

## Flags
- Note any missing tests, hardcoded secrets/credentials, or leftover TODOs.
- Write "None noticed" if nothing stands out.

## Files
- top 5 changed files

Do not invent changes that are not in the diff."""

SECURITY_SYSTEM_PROMPT = """You are a security reviewer analyzing a code diff for a pull request.
You receive the diff plus findings from a heuristic security scan.

Return ONLY markdown with:

## Security Impact
- 2 to 4 short bullets: which parts of the product this change touches from a
  security perspective (authentication, data handling, network, dependencies,
  CI/CD, configuration) and what the change does to them.

## Harm Assessment
Start with exactly one of: "No harm identified" | "Potential harm" | "Likely harm".
Then 1 to 3 bullets with the concrete attack scenario or weakness this change
introduces, or why it is safe.

## Severity
none | low | medium | high | critical (one word, then one short reason)

## Recommendations
- 1 to 3 concrete fixes or mitigations, one line each. Write "None needed" if the change is safe.

Base the analysis ONLY on the diff and the heuristic findings provided.
Do not invent changes that are not in the diff."""

# (category, severity, compiled regex, message) applied to ADDED lines only,
# so findings always trace back to what this change introduces.
SECURITY_RULES = [
    ("hardcoded-secret", "high",
     re.compile(r"""(?i)\b(api[_-]?key|apikey|secret|token|passw(?:or)?d|passwd)\b\s*[:=]\s*["'][^"']{4,}["']"""),
     "possible hardcoded credential"),
    ("hardcoded-secret", "high",
     re.compile(r"AKIA[0-9A-Z]{16}"),
     "possible AWS access key ID"),
    ("hardcoded-secret", "high",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
     "private key material"),
    ("injection", "high",
     re.compile(r"""(?i)["'].*\b(select|insert\s+into|update|delete\s+from)\b.*["']\s*\+"""),
     "possible SQL injection via string concatenation"),
    ("injection", "high",
     re.compile(r"""(?i)\bf["'].*\b(select|insert\s+into|update|delete\s+from)\b.*\{"""),
     "possible SQL injection via f-string"),
    ("dangerous-call", "high",
     re.compile(r"\beval\s*\(|\bexec\s*\("),
     "eval/exec can run arbitrary code"),
    ("dangerous-call", "high",
     re.compile(r"os\.system\s*\(|shell\s*=\s*True"),
     "shell command execution"),
    ("dangerous-call", "high",
     re.compile(r"pickle\.loads?\s*\("),
     "unpickling untrusted data can execute code"),
    ("dangerous-call", "medium",
     re.compile(r"yaml\.load\s*\((?![^)]*SafeLoader)"),
     "yaml.load without SafeLoader"),
    ("insecure-transport", "medium",
     re.compile(r"verify\s*=\s*False"),
     "TLS certificate verification disabled"),
    ("insecure-transport", "medium",
     re.compile(r"ssl\._create_unverified_context"),
     "unverified SSL context"),
    ("insecure-transport", "medium",
     re.compile(r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)\S"),
     "unencrypted http:// URL"),
    ("weak-crypto", "medium",
     re.compile(r"(?i)\b(md5|sha1)\s*\("),
     "weak hash algorithm"),
    ("weak-crypto", "medium",
     re.compile(r"\bMODE_ECB\b|\bDES\b"),
     "weak cipher or mode"),
    ("risky-config", "medium",
     re.compile(r"(?i)\bdebug\s*=\s*True"),
     "debug mode enabled"),
    ("risky-config", "medium",
     re.compile(r"""(?i)access-control-allow-origin["']?\s*[:=]\s*["']\*|allow_origins\s*=\s*\[\s*["']\*"""),
     "CORS wildcard origin"),
    ("risky-config", "medium",
     re.compile(r"chmod\s+[0-7]*777\b|\b0o777\b"),
     "world-writable permissions"),
]

# Paths whose changes matter for security even when no rule matches a line.
SENSITIVE_PATH_PATTERNS = [
    (re.compile(r"(?i)(auth|login|passw|secret|token|session|crypt|oauth|sso|acl|permission|cert)"),
     "filename suggests security-sensitive code"),
    (re.compile(r"^\.github/workflows/"), "CI/CD pipeline definition"),
    (re.compile(r"(?i)(^|/)dockerfile|docker-compose"), "container build/runtime configuration"),
    (re.compile(r"(^|/)(requirements[^/]*\.txt|package\.json|Pipfile|pyproject\.toml|go\.mod|Gemfile)$"),
     "dependency manifest (supply-chain surface)"),
    (re.compile(r"(^|/)\.env"), "environment/credentials file"),
]

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def is_ignored(diff_line):
    return any(pattern in diff_line for pattern in IGNORED_PATTERNS)


def filter_diff(diff_text):
    lines = diff_text.splitlines(keepends=True)
    kept = []
    skipping = False
    for line in lines:
        if line.startswith("diff --git"):
            skipping = is_ignored(line)
        if not skipping:
            kept.append(line)
    return "".join(kept)


def split_by_file(diff_text):
    """Split a diff into a list of (filename, chunk_text) for each file section."""
    lines = diff_text.splitlines(keepends=True)
    chunks = []
    current_name = None
    current_lines = []
    for line in lines:
        if line.startswith("diff --git"):
            if current_name is not None:
                chunks.append((current_name, "".join(current_lines)))
            current_name = line.strip().split(" b/")[-1]
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_name is not None:
        chunks.append((current_name, "".join(current_lines)))
    return chunks


def smart_truncate(diff_text, max_chars):
    """Keep whole per-file diffs until the budget is used up, instead of cutting mid-file."""
    chunks = split_by_file(diff_text)
    if not chunks:
        return diff_text[:max_chars], []

    kept = []
    omitted = []
    used = 0
    for name, chunk in chunks:
        if used + len(chunk) <= max_chars:
            kept.append(chunk)
            used += len(chunk)
        else:
            omitted.append(name)

    if not kept:
        return diff_text[:max_chars], [name for name, _ in chunks]

    return "".join(kept), omitted


def scan_security(diff_text):
    """Scan ADDED lines of a diff for security-relevant patterns.

    Returns a list of findings sorted by severity:
    {"severity", "category", "file", "line", "message", "evidence"}
    Path-based findings (security-sensitive files touched) have line=None.
    """
    findings = []
    current_file = None
    new_line = None
    flagged_paths = set()

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            current_file = line.strip().split(" b/")[-1]
            new_line = None
            if current_file not in flagged_paths:
                for pattern, reason in SENSITIVE_PATH_PATTERNS:
                    if pattern.search(current_file):
                        flagged_paths.add(current_file)
                        findings.append({
                            "severity": "low",
                            "category": "sensitive-file",
                            "file": current_file,
                            "line": None,
                            "message": reason,
                            "evidence": "",
                        })
                        break
            continue
        hunk = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
        if hunk:
            new_line = int(hunk.group(1))
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            content = line[1:]
            for category, severity, pattern, message in SECURITY_RULES:
                if pattern.search(content):
                    findings.append({
                        "severity": severity,
                        "category": category,
                        "file": current_file or "unknown",
                        "line": new_line,
                        "message": message,
                        "evidence": content.strip()[:90],
                    })
            if new_line is not None:
                new_line += 1
        elif not line.startswith("-"):
            if new_line is not None:
                new_line += 1

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 3))
    return findings


def format_security_report(findings):
    lines = ["## Security Findings (heuristic scan)"]
    code_findings = [f for f in findings if f["category"] != "sensitive-file"]
    path_findings = [f for f in findings if f["category"] == "sensitive-file"]

    if code_findings:
        for f in code_findings:
            location = f["file"] if f["line"] is None else f"{f['file']}:{f['line']}"
            lines.append(f"- [{f['severity']}] {location} — {f['category']}: {f['message']}")
            if f["evidence"]:
                lines.append(f"    `{f['evidence']}`")
    else:
        lines.append("- None detected. (Heuristics only — not a full security audit.)")

    if path_findings:
        lines.append("")
        lines.append("## Security-Sensitive Areas Touched")
        for f in path_findings:
            lines.append(f"- {f['file']} — {f['message']}")

    return "\n".join(lines)


def read_input(file_path, base_branch):
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    if base_branch:
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...HEAD"], capture_output=True, text=True
        )
    else:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    return result.stdout


def call_llm(client, model, diff_text, system_prompt=SYSTEM_PROMPT):
    max_retries = 3
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": diff_text},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"Attempt {attempt} failed ({e}). Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"Could not generate summary after {max_retries} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Summarize a git diff for a PR.")
    parser.add_argument("--file", help="Path to a .diff file. If omitted, reads from stdin or runs git diff.")
    parser.add_argument("--base", help="Base branch to diff against (e.g. main). Only used when no --file/stdin.")
    parser.add_argument("--dry-run", action="store_true", help="Show the filtered diff without calling the LLM.")
    parser.add_argument("--security", action="store_true",
                        help="Security-focused analysis: heuristic vulnerability scan plus an LLM "
                             "assessment of security impact and potential harm to the product.")
    args = parser.parse_args()

    diff_text = read_input(args.file, args.base)

    if not diff_text or not diff_text.strip():
        print("Error: input diff is empty.", file=sys.stderr)
        sys.exit(1)

    diff_text = filter_diff(diff_text)

    if not diff_text.strip():
        print("Error: diff only contained ignored files (lockfiles, minified assets).", file=sys.stderr)
        sys.exit(1)

    diff_text, omitted = smart_truncate(diff_text, MAX_CHARS)

    if omitted:
        print(f"(Note: {len(omitted)} file(s) omitted to fit size limit: {', '.join(omitted)})", file=sys.stderr)

    findings = scan_security(diff_text)

    if args.security:
        report = format_security_report(findings)
        print(report)
        if args.dry_run:
            return
    elif findings:
        print(
            f"(Security: {len(findings)} potential issue(s) detected — "
            "run again with --security for a security-focused analysis.)",
            file=sys.stderr,
        )

    if args.dry_run:
        print(diff_text)
        return

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL", "gpt-oss:20b-cloud")

    if not api_key:
        print("Error: LLM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    if args.security:
        system_prompt = SECURITY_SYSTEM_PROMPT
        user_content = (
            "HEURISTIC SCAN FINDINGS:\n" + format_security_report(findings)
            + "\n\nDIFF:\n" + diff_text
        )
    else:
        system_prompt = SYSTEM_PROMPT
        user_content = diff_text

    try:
        summary = call_llm(client, model, user_content, system_prompt=system_prompt)
        print(summary)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
