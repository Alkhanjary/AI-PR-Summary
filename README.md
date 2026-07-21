# AI PR Summary
## Installation

### 1. Install Git Bash (if you don't have it)

Download and install Git for Windows from https://git-scm.com/downloads — this includes Git Bash.

### 2. Get the exact tool files onto your computer

Open Git Bash, then clone this repo (this pulls the real, working files — not a recreation):

cd ~/desktop
git clone https://github.com/Alkhanjary/AI-PR-Summary.git
cd AI-PR-Summary

### 3. Open Git Bash in any other folder later

Right-click inside any folder in Windows Explorer -> "Git Bash Here". Or from an already-open Git Bash window:

cd /c/Users/YourWindowsUsername/path/to/folder

### 4. Continue with Quick Start below

## Quick Start

py -m pip install -r requirements.txt
cp .env.example .env
# edit .env with your real LLM_API_KEY

echo "alias prsum='py /full/path/to/AI-PR-Summary/summarize_pr.py'" >> ~/.bashrc
source ~/.bashrc

git diff | prsum

---


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

**Summarize your whole branch, compared to main:**

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

## 3. Security Analysis Mode (--security)

Adds a cybersecurity-focused review of the change: what the change impacts in
the product from a security perspective, and whether it could harm the code.

**Full security analysis (heuristic scan + LLM assessment):**

git diff | prsum --security

**Offline scan only — no API call, no key needed:**

git diff | prsum --security --dry-run

**Machine-readable output + merge gate (for CI):**

git diff | prsum --json --fail-on high

`--json` prints findings as JSON (never calls the LLM). `--fail-on high`
makes the command exit with code **2** when findings at or above that
severity exist — that exit code is the merge gate. Exit codes: 0 = clean,
1 = operational error, 2 = findings at/above the threshold.

### What it does

1. **Heuristic scan (deterministic, runs offline).** Every *added* line of the
   **complete raw diff** is checked against known risky patterns — the scan runs
   *before* lockfile filtering and size truncation, so a large or noisy diff
   cannot hide a finding. Patterns covered:
   - hardcoded secrets (quoted or unquoted assignments, JSON keys, AWS keys,
     private keys) — obvious placeholders like `your-api-key` are skipped, and
     detected secret values are **redacted** from all output
   - injection risks (SQL built by concatenation, f-strings, or `$var` interpolation)
   - dangerous calls (eval/exec, os.system, shell=True, pickle.loads, unsafe yaml.load)
   - XSS sinks (innerHTML, document.write, dangerouslySetInnerHTML)
   - GitHub Actions expression injection (untrusted event data in `${{ }}`)
   - insecure transport (verify=False, unverified SSL context, plain http:// URLs)
   - weak crypto (MD5/SHA1 hashing, DES/ECB)
   - risky config (debug=True, CORS wildcard, chmod 777)

   It also flags when the change touches **security-sensitive areas**: auth/
   login/token/crypto files, CI/CD workflows, Dockerfiles, dependency
   manifests, and .env files.

2. **LLM assessment.** The diff plus the scan findings are sent to the LLM
   with a security-reviewer prompt that returns:
   - **Security Impact** — which parts of the product the change touches
   - **Harm Assessment** — "No harm identified" / "Potential harm" / "Likely harm", with the concrete attack scenario
   - **Severity** — none | low | medium | high | critical
   - **Recommendations** — concrete fixes or mitigations

### Example output

## Security Findings (heuristic scan)
- [high] auth.py:2 — injection: possible SQL injection via string concatenation
    `query = "SELECT * FROM users WHERE name = '" + user + "'"`
- [high] auth.py:3 — hardcoded-secret: possible hardcoded credential
    `PASSWORD = [REDACTED]`

## Security-Sensitive Areas Touched
- auth.py — filename suggests security-sensitive code

(...followed by the LLM's Security Impact / Harm Assessment / Severity / Recommendations sections.)

In normal mode (without --security), the tool still runs the scan quietly and
prints a one-line note to stderr when it spots something, so risky changes
never pass completely silently.

### Positioning and limitations

- **The gate is deterministic; the LLM is advisory.** Only the regex scan's
  exit code can block a merge. The LLM assessment is never used as a gate,
  because PR diffs are attacker-controlled input and could attempt prompt
  injection (both prompts also instruct the model to ignore embedded
  instructions).
- **This is a fast tripwire, not a security audit.** The rules catch common,
  obvious mistakes. For depth, layer dedicated tools on top: secret scanning
  (Gitleaks/TruffleHog), SAST (Semgrep/CodeQL), and dependency/vulnerability
  scanning (OSV-Scanner/Trivy/Dependabot).
- Findings are heuristic: expect some false positives (a matched pattern in a
  comment) and false negatives (novel or language-specific issues). Treat
  every finding as a pointer for human review.

---

## 4. Automatic Setup (GitHub Action, any repo)

Instead of running the tool yourself, GitHub can run it automatically on every pull request and post the summary as a PR comment.

### Step 1 — Add secrets to the target repo

In the target repo, go to:
Settings -> Secrets and variables -> Actions -> New repository secret

Add:
- LLM_API_KEY
- LLM_BASE_URL -> https://ollama.com/v1
- LLM_MODEL -> gpt-oss:20b-cloud

### Step 2 — Add a workflow file to that repo

Create .github/workflows/pr-summary.yml in the target repo with this content (replace YOUR_GITHUB_USERNAME):

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
    with:
      fail_on: high   # optional — block merge on high findings (high | medium | low | none)
    secrets:
      LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}
      LLM_MODEL: ${{ secrets.LLM_MODEL }}

The real logic lives in this repo's reusable-pr-summary.yml. Every repo pointing to it shares the same behavior, and improving it here updates it everywhere on the next PR. The workflow checks out this repo and runs the same summarize_pr.py used locally — there is no duplicated logic.

### What happens on every PR

- Diffs the PR branch against its base branch (full raw diff)
- **Runs the deterministic security scan on the complete diff** and uploads
  the findings as a machine-readable security-findings JSON artifact
- Sends the (filtered, truncated) diff to the LLM for the advisory summary
- Posts one comment containing the summary plus the security findings
  (updates the same comment on new pushes, no spam)
- **Fails the check** if findings at or above the fail_on threshold exist
  (default: high) — or if the scan itself errored (fail-closed). Make the
  check required in branch protection to actually block merging.

---

## Notes

- Diffs over ~8,000 characters are truncated automatically.
- Lockfiles and minified assets are excluded from what's sent to the LLM.
- Empty input is rejected with a clear error instead of wasting an API call.
- On API failure, the tool prints a clear error instead of crashing silently.

