# AI PR Summary

Generate a short PR summary (bullets + risk + top files) from a git diff, using an LLM.

## Setup

1. Install dependencies:

py -m pip install -r requirements.txt

2. Copy .env.example to .env and fill in your values:

LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://ollama.com/v1
LLM_MODEL=gpt-oss:20b-cloud

## Usage

git diff main...HEAD > change.diff
py summarize_pr.py --file change.diff

# or, pipe it directly:
git diff main...HEAD | py summarize_pr.py

## Notes

- Diffs larger than ~8,000 characters are truncated automatically.
- Empty input is rejected with a clear error.
- On API/network failure, the script prints "Could not generate summary" and exits non-zero.
