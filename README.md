# AI PR Summary

A tool that reads a git diff and generates a PR-ready summary using an LLM (via Ollama Cloud) — including a suggested title, what changed, what it does, how risky it is, and anything worth flagging (missing tests, secrets, TODOs).

Works two ways:
- **Locally**, as a command you run yourself before opening a PR
- **Automatically**, as a GitHub Action that comments on every PR in any repo you add it to

---

## 1. Local Setup

### Install dependencies

py -m pip install -r requirements.txt

### Configure your API key

Copy .env.example to .env:

cp .env.example .env

Then edit .env with your real values:

LLM_API_KEY=your-ollama-cloud-api-key
LLM_BASE_URL=https://ollama.com/v1
LLM_MODEL=gpt-oss:20b-cloud

.env is git-ignored — your key never gets committed.

### (Optional but recommended) Set up the prsum shortcut

So you can run the tool from any repo on your machine, not just this folder:

echo "alias prsum='py /full/path/to/AI-PR-Summary/summarize_pr.py'" >> ~/.bashrc
source ~/.bashrc

(Replace the path with wherever you cloned this repo.)

---

## 2. Local Usage

The core idea: git diff produces the changes, the tool reads them and returns a summary.

**Most common — summarize your current uncommitted changes:**

git diff | prsum

**Summarize your whole branch, compared to main (what you'd typically paste into a PR description):**

git diff main...HEAD | prsum

**From a saved diff file:**

git diff > change.diff
prsum --file change.diff

### Example output

## Suggested Title
fix: handle empty diff input gracefully

## Summary
- Added a check for empty diff before calling the API
- Improved error message on API failure

## Impact
- Prevents wasted API calls on empty input
- Easier to debug failures from the CLI

## Risk
low | error handling only, no behavior change

## Flags
- None noticed

## Files
- summarize_pr.py

Copy that output straight into your PR description.

---

## 3. Automatic Setup (GitHub Action, any repo)

Instead of running the tool yourself, you can have GitHub run it automatically on every pull request, and post the summary as a PR comment.

### Step 1 — Add secrets to the target repo

In the repo you want this on (not this one), go to:
Settings -> Secrets and variables -> Actions -> New repository secret

Add:
- LLM_API_KEY
- LLM_BASE_URL -> https://ollama.com/v1
- LLM_MODEL -> gpt-oss:20b-cloud

### Step 2 — Add a small workflow file to that repo

Create .github/workflows/pr-summary.yml in the target repo:

name: AI PR Summary

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  summarize:
    permissions:
      pull-requests: write
      contents: read
      issues: write
    uses: YOUR_GITHUB_USERNAME/AI-PR-Summary/.github/workflows/reusable-pr-summary.yml@main
    secrets:
      LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
      LLM_MODEL: ${{ secrets.LLM_MODEL }}

Replace YOUR_GITHUB_USERNAME with your actual GitHub username.

That's it. The real logic lives in this repo's reusable-pr-summary.yml — every repo pointing to it shares the same behavior, and improving it here updates it everywhere automatically on the next PR.

### What happens on every PR

- The workflow diffs the PR branch against its base branch
- Sends it to the LLM
- Posts a comment on the PR with the summary (updates the same comment on new pushes, doesn't spam)

---

## Notes

- Diffs over ~8,000 characters are truncated automatically.
- Lockfiles and minified assets (package-lock.json,
cat > README.md << 'EOF'
# AI PR Summary

A tool that reads a git diff and generates a PR-ready summary using an LLM (via Ollama Cloud) — including a suggested title, what changed, what it does, how risky it is, and anything worth flagging (missing tests, secrets, TODOs).

Works two ways:
- **Locally**, as a command you run yourself before opening a PR
- **Automatically**, as a GitHub Action that comments on every PR in any repo you add it to

---

## 1. Local Setup

### Install dependencies

py -m pip install -r requirements.txt

### Configure your API key

Copy .env.example to .env:

cp .env.example .env

Then edit .env with your real values:

LLM_API_KEY=your-ollama-cloud-api-key
LLM_BASE_URL=https://ollama.com/v1
LLM_MODEL=gpt-oss:20b-cloud

.env is git-ignored — your key never gets committed.

### (Optional but recommended) Set up the prsum shortcut

So you can run the tool from any repo on your machine, not just this folder:

echo "alias prsum='py /full/path/to/AI-PR-Summary/summarize_pr.py'" >> ~/.bashrc
source ~/.bashrc

(Replace the path with wherever you cloned this repo.)

---

## 2. Local Usage

The core idea: git diff produces the changes, the tool reads them and returns a summary.

**Most common — summarize your current uncommitted changes:**

git diff | prsum

**Summarize your whole branch, compared to main (what you'd typically paste into a PR description):**

git diff main...HEAD | prsum

**From a saved diff file:**

git diff > change.diff
prsum --file change.diff

### Example output

## Suggested Title
fix: handle empty diff input gracefully

## Summary
- Added a check for empty diff before calling the API
- Improved error message on API failure

## Impact
- Prevents wasted API calls on empty input
- Easier to debug failures from the CLI

## Risk
low | error handling only, no behavior change

## Flags
- None noticed

## Files
- summarize_pr.py

Copy that output straight into your PR description.

---

## 3. Automatic Setup (GitHub Action, any repo)

Instead of running the tool yourself, you can have GitHub run it automatically on every pull request, and post the summary as a PR comment.

### Step 1 — Add secrets to the target repo

In the repo you want this on (not this one), go to:
Settings -> Secrets and variables -> Actions -> New repository secret

Add:
- LLM_API_KEY
- LLM_BASE_URL -> https://ollama.com/v1
- LLM_MODEL -> gpt-oss:20b-cloud

### Step 2 — Add a small workflow file to that repo

Create .github/workflows/pr-summary.yml in the target repo:

name: AI PR Summary

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  summarize:
    permissions:
      pull-requests: write
      contents: read
      issues: write
    uses: YOUR_GITHUB_USERNAME/AI-PR-Summary/.github/workflows/reusable-pr-summary.yml@main
    secrets:
      LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
      LLM_MODEL: ${{ secrets.LLM_MODEL }}

Replace YOUR_GITHUB_USERNAME with your actual GitHub username.

That's it. The real logic lives in this repo's reusable-pr-summary.yml — every repo pointing to it shares the same behavior, and improving it here updates it everywhere automatically on the next PR.

### What happens on every PR

- The workflow diffs the PR branch against its base branch
- Sends it to the LLM
- Posts a comment on the PR with the summary (updates the same comment on new pushes, doesn't spam)

---

## Notes

- Diffs over ~8,000 characters are truncated automatically.
- Lockfiles and minified assets (package-lock.json, *.min.js, etc.) are excluded from what gets sent to the LLM.
- Empty input is rejected with a clear error instead of wasting an API call.
- On API failure, the tool prints a clear error instead of crashing silently.
