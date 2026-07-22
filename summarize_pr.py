import argparse
import os
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
- Note any hardcoded secrets/credentials or leftover TODOs.
- Write "None noticed" if nothing stands out.

## Suggested Tests
- List specific test file paths to add or update (e.g. `tests/test_auth.py`), each with a short reason tied to what changed in the diff.
- If the diff shows an existing test file or a `tests/` folder convention, follow that naming pattern; otherwise default to `tests/test_<module>.py`.
- Write "None needed" if the change has no testable logic (docs, comments, config-only, formatting).

## Files
- top 5 changed files

Do not invent changes that are not in the diff."""


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


def call_llm(client, model, diff_text):
    max_retries = 3
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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

    try:
        summary = call_llm(client, model, diff_text)
        print(summary)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
