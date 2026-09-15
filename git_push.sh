#!/bin/bash

# =============================================================================
# Git Push Script for the tt1lWcb (W->cb) analysis code on lxplus9
# =============================================================================
# Makes THIS directory its own standalone git repo (independent of the parent
# WWW repo), adds every file in the tree that is not excluded by .gitignore,
# commits, and pushes to BOTH GitLab (CERN) and GitHub.
#
# Usage: ./git_push.sh "Your commit message"
#        ./git_push.sh                 (default message with timestamp)
#
# Auth:
#   GitLab -- Kerberos.  Run `kinit agapitos@CERN.CH` first if the push asks
#             for a password.
#   GitHub -- SSH key (git@github.com:ph2632/tt1lWcb.git).  Verified working
#             on this machine (2026-09-15): `ssh -T git@github.com` auths as
#             user ph2632. The repo must already exist and be empty/related
#             on GitHub -- create it at https://github.com/new (name
#             "tt1lWcb", do NOT initialise with a README/license/.gitignore,
#             or the first push will be rejected as non-fast-forward) BEFORE
#             running this script for the first time.
#
# One commit, pushed to both remotes. A failure on one remote does not skip
# the other (2026-09-15) -- the script's own exit code is nonzero if EITHER
# push failed, so calling code can still detect trouble.
# =============================================================================

set -e  # exit on error (still used for the setup steps before the pushes)

# --- always operate on the directory this script lives in ---------------------
cd "$(dirname "$(readlink -f "$0")")"
REPO_DIR="$(pwd)"

# GitLab repository URL (Kerberos auth for lxplus, port 8443)
GITLAB_REPO="https://:@gitlab.cern.ch:8443/agapitos/tt1lWcb.git"
# GitHub repository URL (SSH key auth -- see the Auth note above)
GITHUB_REPO="git@github.com:ph2632/tt1lWcb.git"
BRANCH="master"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo -e "${GREEN}=== Git Push Script  (${REPO_DIR}) ===${NC}"

# Commit message
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    COMMIT_MSG="Update tt1lWcb analysis code - $(date '+%Y-%m-%d %H:%M')"
fi

# -----------------------------------------------------------------------------
# Step 1: make sure THIS dir is its own git repo.
#   The parent (.../WWW) is also a git repo, so a missing/!broken .git here
#   would make `git add` operate on the parent.  We insist on a real local
#   .git (one that has a HEAD file); a leftover empty skeleton is wiped.
# -----------------------------------------------------------------------------
if [ ! -f .git/HEAD ]; then
    if [ -d .git ]; then
        echo -e "${YELLOW}Removing incomplete .git skeleton...${NC}"
        rm -rf .git
    fi
    echo -e "${YELLOW}Initializing a fresh git repository here...${NC}"
    git init -b "$BRANCH" 2>/dev/null || { git init && git checkout -b "$BRANCH" 2>/dev/null || true; }
fi

# hard safety check: the repo root must be this directory, never the parent
TOPLEVEL="$(git rev-parse --show-toplevel)"
if [ "$TOPLEVEL" != "$REPO_DIR" ]; then
    echo -e "${RED}ABORT: git root is '$TOPLEVEL', not '$REPO_DIR'.${NC}"
    echo -e "${RED}Refusing to push the parent repo. Check the .git directory here.${NC}"
    exit 1
fi

# remotes: "origin" = GitLab (unchanged), "github" = GitHub (new, 2026-09-15)
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
# Step 2: .gitignore  (GitLab rejects files > 100 MB; keep the repo to code)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Writing .gitignore...${NC}"
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

# ---- generated plots / outputs -----------------------------------------
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
# Step 3: stage
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Staging files (respecting .gitignore)...${NC}"
git add -A

# drop any AFS temp / previously-tracked junk that slipped in
git rm -r --cached --ignore-unmatch '.__afs*' '#*#' > /dev/null 2>&1 || true

echo -e "${YELLOW}Staged:${NC}"
git status --short

# guard: warn on any staged file > 5 MB
echo -e "${YELLOW}Checking for large staged files...${NC}"
git diff --cached --name-only --diff-filter=ACM | while read -r f; do
    [ -f "$f" ] || continue
    kb=$(du -k "$f" | cut -f1)
    if [ "$kb" -gt 5120 ]; then
        echo -e "${RED}  WARNING: large file staged: $f  (${kb} kB) — add it to .gitignore${NC}"
    fi
done

# -----------------------------------------------------------------------------
# Step 4: commit + push
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Committing: ${COMMIT_MSG}${NC}"
git commit -m "$COMMIT_MSG" || echo -e "${YELLOW}Nothing to commit (working tree clean)${NC}"

# set -e is relaxed here on purpose: a failure pushing to ONE remote must
# not skip the other (2026-09-15) -- each push is checked explicitly instead,
# and the script's own exit code reflects whether either one failed.
set +e
FAILED=0

echo -e "${YELLOW}Pushing to origin/${BRANCH} (GitLab) ...${NC}"
git push -u origin "$BRANCH" || git push --set-upstream origin "$BRANCH"
if [ $? -ne 0 ]; then
    echo -e "${RED}  GitLab push FAILED.${NC}"
    FAILED=1
else
    echo -e "${GREEN}  GitLab push OK.${NC}"
fi

echo -e "${YELLOW}Pushing to github/${BRANCH} (GitHub) ...${NC}"
if git push -u github "$BRANCH" 2>&1 | tee /tmp/git_push_github.$$.log; then
    echo -e "${GREEN}  GitHub push OK.${NC}"
else
    echo -e "${RED}  GitHub push FAILED.${NC}"
    if grep -qi "repository not found" /tmp/git_push_github.$$.log; then
        echo -e "${RED}  -> the repo does not exist yet (or the SSH key has no access).${NC}"
        echo -e "${RED}     Create an EMPTY repo named 'tt1lWcb' at https://github.com/new${NC}"
        echo -e "${RED}     (owner: ph2632; do NOT add a README/license/.gitignore there),${NC}"
        echo -e "${RED}     then re-run this script.${NC}"
    elif grep -qi "non-fast-forward\|fetch first\|rejected" /tmp/git_push_github.$$.log; then
        echo -e "${RED}  -> the GitHub repo has commits this local repo doesn't (e.g. it was${NC}"
        echo -e "${RED}     initialised with a README). Resolve manually, e.g.:${NC}"
        echo -e "${RED}       git fetch github && git merge --allow-unrelated-histories github/${BRANCH}${NC}"
    fi
    FAILED=1
fi
rm -f /tmp/git_push_github.$$.log

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}=== Done! Pushed to GitLab and GitHub. ===${NC}"
else
    echo -e "${RED}=== Done, but at least one remote push FAILED -- see above. ===${NC}"
fi
exit "$FAILED"
