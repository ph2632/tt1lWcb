#!/bin/bash

# =============================================================================
# Git Push Script for the tt1lWcb (W->cb) analysis code on lxplus9
# =============================================================================
# Makes THIS directory its own standalone git repo (independent of the parent
# WWW repo), adds every file in the tree that is not excluded by .gitignore,
# commits, and pushes to BOTH GitLab (CERN) and GitHub. Code only -- .gitignore
# excludes generated outputs (*.png/*.pdf/*.root/*.npz/... -- see below).
#
# Usage: ./git_push.sh "Your commit message"
#        ./git_push.sh                 (default message with timestamp)
#
# Auth:
#   GitLab -- Kerberos.  Run `kinit agapitos@CERN.CH` first if the push asks
#             for a password.
#   GitHub -- SSH key (git@github.com:ph2632/tt1lWcb.git).  Verified working
#             on this machine (2026-09-15): `ssh -T git@github.com` auths as
#             user ph2632. The repo must already exist on GitHub (create it
#             empty at https://github.com/new, name "tt1lWcb", no README/
#             license/.gitignore) before running this script for the first
#             time.
#
# One commit, pushed to both remotes. A failure on one remote does not skip
# the other; the script's own exit code is nonzero if EITHER push failed.
#
# Output (2026-09-15, user: "minimum verbosity but reporting the file
# changes"): quiet by default -- the only routine output is the list of
# changed files (git status --short) and a one-line OK/FAILED per remote.
# Setup actions (first-time repo init, adding a remote) still print one line
# since those only ever happen once. On a push failure the real git error
# text is shown, followed by a diagnosis.
# =============================================================================

set -e  # exit on error (setup steps only -- relaxed before the pushes, see below)

# --- always operate on the directory this script lives in ---------------------
cd "$(dirname "$(readlink -f "$0")")"
REPO_DIR="$(pwd)"

# GitLab repository URL (Kerberos auth for lxplus, port 8443)
GITLAB_REPO="https://:@gitlab.cern.ch:8443/agapitos/tt1lWcb.git"
# GitHub repository URL (SSH key auth -- see the Auth note above)
GITHUB_REPO="git@github.com:ph2632/tt1lWcb.git"
BRANCH="master"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# Commit message
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    COMMIT_MSG="Update tt1lWcb analysis code - $(date '+%Y-%m-%d %H:%M')"
fi

# -----------------------------------------------------------------------------
# Step 1: make sure THIS dir is its own git repo (silent unless first-time).
#   The parent (.../WWW) is also a git repo, so a missing/broken .git here
#   would make `git add` operate on the parent.  We insist on a real local
#   .git (one that has a HEAD file); a leftover empty skeleton is wiped.
# -----------------------------------------------------------------------------
if [ ! -f .git/HEAD ]; then
    [ -d .git ] && rm -rf .git
    git init -b "$BRANCH" >/dev/null 2>&1 \
        || { git init >/dev/null && git checkout -b "$BRANCH" >/dev/null 2>&1 || true; }
    echo -e "${YELLOW}Initialized fresh git repository here.${NC}"
fi

# hard safety check: the repo root must be this directory, never the parent
TOPLEVEL="$(git rev-parse --show-toplevel)"
if [ "$TOPLEVEL" != "$REPO_DIR" ]; then
    echo -e "${RED}ABORT: git root is '$TOPLEVEL', not '$REPO_DIR'.${NC}"
    echo -e "${RED}Refusing to push the parent repo. Check the .git directory here.${NC}"
    exit 1
fi

# remotes: "origin" = GitLab, "github" = GitHub (silent unless newly added)
if ! git remote | grep -q '^origin$'; then
    git remote add origin "$GITLAB_REPO"
    echo -e "${YELLOW}Added remote origin -> ${GITLAB_REPO}${NC}"
else
    git remote set-url origin "$GITLAB_REPO"
fi
if ! git remote | grep -q '^github$'; then
    git remote add github "$GITHUB_REPO"
    echo -e "${YELLOW}Added remote github -> ${GITHUB_REPO}${NC}"
else
    git remote set-url github "$GITHUB_REPO"
fi

# -----------------------------------------------------------------------------
# Step 2: .gitignore (silent write every run -- GitLab rejects files > 100 MB,
# and per 2026-09-15 user instruction this repo is CODE ONLY: no generated
# plots (*.png/*.pdf/...) or data/model blobs are ever tracked).
# -----------------------------------------------------------------------------
cat > .gitignore << 'EOF'
# ---- environments / caches -------------------------------------------------
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.claude/

# ---- EOS symlinks (machine-specific, point outside the repo) --------------
bjet_cache
onnx
cache_batch_parquet
cache_shape_comparison
score_parquet
score_parquet_flatten
score_parquet_swqq_bdt

# ---- big data / model formats -------------------------------------------
*.root
*.pkl
*.pickle
*.npz
*.npy
*.h5
*.hdf5
*.parquet
*.onnx
*.pb

# ---- generated plots / outputs -- CODE ONLY in this repo (2026-09-15) ---
*.png
*.pdf
*.eps
*.svg
figures/
figures_swqq_bdt/
debug_figures/
plots/
shape_comparison/
parquet/

# ---- local scratch / backups -----------------------------------------
fake_data/
bak/
BU/
BU_*/
*.bak
*.bak_*
*.orig
\#*\#
*~
*.swp
*.swo
*.log

# ---- OS cruft -----------------------------------------------------------
.DS_Store
._*
.__afs*
EOF

# -----------------------------------------------------------------------------
# Step 3: stage, report the file changes (the one thing this script always
# prints), guard against large files slipping past .gitignore.
# -----------------------------------------------------------------------------
git add -A
git rm -r --cached --ignore-unmatch '.__afs*' '#*#' > /dev/null 2>&1 || true

CHANGES="$(git status --short)"
if [ -n "$CHANGES" ]; then
    echo "$CHANGES"
else
    echo "(no file changes)"
fi

git diff --cached --name-only --diff-filter=ACM | while read -r f; do
    [ -f "$f" ] || continue
    kb=$(du -k "$f" | cut -f1)
    if [ "$kb" -gt 5120 ]; then
        echo -e "${RED}  WARNING: large file staged: $f  (${kb} kB) -- add it to .gitignore${NC}"
    fi
done

# -----------------------------------------------------------------------------
# Step 4: commit + push (quiet -- errors still surface in full)
# -----------------------------------------------------------------------------
if git commit -q -m "$COMMIT_MSG" > /dev/null 2>&1; then
    echo -e "${GREEN}Committed:${NC} ${COMMIT_MSG}"
else
    echo -e "${YELLOW}Nothing to commit.${NC}"
fi

# set -e relaxed here on purpose: a failure pushing to ONE remote must not
# skip the other -- each push is checked explicitly, and the script's own
# exit code reflects whether either one failed.
set +e
FAILED=0
LOG="/tmp/git_push.$$.log"
SEP="---------------------------------------"

echo "$SEP"
echo -e "${YELLOW}GitLab${NC}  (origin -> gitlab.cern.ch/agapitos/tt1lWcb)"
git push -q -u origin "$BRANCH" 2>"$LOG"
if [ $? -ne 0 ]; then
    echo -e "${RED}GitLab push FAILED:${NC}"
    cat "$LOG"
    FAILED=1
else
    echo -e "${GREEN}GitLab: OK${NC}"
fi

echo "$SEP"
echo -e "${YELLOW}GitHub${NC}  (github -> github.com/ph2632/tt1lWcb)"
git push -q -u github "$BRANCH" 2>"$LOG"
gh_status=$?

# 2026-09-15, user: "tune the output to avoid false alarm messages" -- a
# non-zero push exit is not always a real failure (GitHub's own ref-lock
# race can report "rejected" for a commit that landed anyway, seen once).
# SELF-VERIFY instead of guessing from the error text: ask GitHub directly
# what commit its branch is actually at and compare to local HEAD. If they
# already match, the push demonstrably succeeded.
if [ "$gh_status" -ne 0 ]; then
    remote_sha="$(git ls-remote github "$BRANCH" 2>/dev/null | cut -f1)"
    local_sha="$(git rev-parse HEAD)"
    if [ -n "$remote_sha" ] && [ "$remote_sha" = "$local_sha" ]; then
        gh_status=0
        echo -e "${YELLOW}(GitHub reported an error, but github/${BRANCH} already matches local HEAD -- treating as success.)${NC}"
    fi
fi

if [ "$gh_status" -eq 0 ]; then
    echo -e "${GREEN}GitHub: OK${NC}"
else
    echo -e "${RED}GitHub push FAILED:${NC}"
    cat "$LOG"
    if grep -qi "repository not found" "$LOG"; then
        echo -e "${RED}-> repo doesn't exist yet (or the SSH key has no access).${NC}"
        echo -e "${RED}   Create an EMPTY repo named 'tt1lWcb' at https://github.com/new${NC}"
        echo -e "${RED}   (owner: ph2632; no README/license/.gitignore there), then re-run.${NC}"
    elif grep -qi "cannot lock ref" "$LOG"; then
        # confirmed a REAL mismatch above (self-verify didn't clear it) --
        # a genuine second writer racing this push, not just a report glitch.
        echo -e "${RED}-> ref-lock conflict and github/${BRANCH} does NOT match local HEAD --${NC}"
        echo -e "${RED}   something else pushed there concurrently. Just re-run this script.${NC}"
    elif grep -qi "non-fast-forward\|fetch first\|rejected" "$LOG"; then
        echo -e "${RED}-> the GitHub repo has commits this local repo doesn't (e.g. it was${NC}"
        echo -e "${RED}   initialised with a README). Resolve manually, e.g.:${NC}"
        echo -e "${RED}     git fetch github && git merge --allow-unrelated-histories github/${BRANCH}${NC}"
    fi
    FAILED=1
fi
rm -f "$LOG"
echo "$SEP"

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}Done -- pushed to GitLab and GitHub.${NC}"
else
    echo -e "${RED}Done, but at least one remote push FAILED -- see above.${NC}"
fi
exit "$FAILED"
