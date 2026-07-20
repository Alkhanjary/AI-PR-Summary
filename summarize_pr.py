import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

MAX_CHARS = 8000

SYSTEM_PROMPT = """You summarize code diffs for PR reviewers.

Return ONLY markdown with:

## Summary
- 2 to 4 short bullets of what changed

## Risk
low | medium | high (one word, then one short reason)

## Files
- top 5 changed files

Do not invent changes that are not in the diff."""


def read_input(file_path):
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def main():
    parser = argparse.ArgumentParser(description="Summarize a git diff for a PR.")
    parser.add_argument("--file", help="Path to a .diff file. If omitted, reads from stdin.")
    args = parser.parse_args()

    diff_text = read_input(args.file)

    if not diff_text or not diff_text.strip():
        print("Error: input diff is empty.", file=sys.stderr)
        sys.exit(1)

    truncated = False
    if len(diff_text) > MAX_CHARS:
        diff_text = diff_text[:MAX_CHARS]
        truncated = True

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or None

    if not api_key:
        print("Error: LLM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    if truncated:
        print(f"(Note: diff truncated to {MAX_CHARS} characters. Consider splitting large PRs.)", file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-oss:20b-cloud"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": diff_text},
            ],
        )
        print(response.choices[0].message.content)
    except Exception:
        print("Could not generate summary", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
