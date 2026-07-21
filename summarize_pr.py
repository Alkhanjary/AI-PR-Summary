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
- Note any missing tests, hardcoded secrets/credentials, or leftover TODOs.
- Write "None noticed" if nothing stands out.

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


def read_input(file_path):
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    result = subprocess.run(["git", "diff"], capture_output=True, text=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Summarize a git diff for a PR.")
    parser.add_argument("--file", help="Path to a .diff file. If omitted, reads from stdin or runs git diff.")
    args = parser.parse_args()

    diff_text = read_input(args.file)

    if not diff_text or not diff_text.strip():
        print("Error: input diff is empty.", file=sys.stderr)
        sys.exit(1)

    diff_text = filter_diff(diff_text)

    if not diff_text.strip():
        print("Error: diff only contained ignored files (lockfiles, minified assets).", file=sys.stderr)
        sys.exit(1)

    truncated = False
    if len(diff_text) > MAX_CHARS:
        diff_text = diff_text[:MAX_CHARS]
        truncated = True

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None
    model = os.environ.get("LLM_MODEL", "gpt-oss:20b-cloud")

    if not api_key:
        print("Error: LLM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    if truncated:
        print(f"(Note: diff truncated to {MAX_CHARS} characters)", file=sys.stderr)

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
            print(response.choices[0].message.content)
            return
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"Attempt {attempt} failed ({e}). Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)

    print(f"Could not generate summary after {max_retries} attempts: {last_error}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
