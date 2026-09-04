#!/bin/bash

# =============================================================================
# Git Push Script for the tt1lWcb (W->cb) analysis code on lxplus9
# =============================================================================
# Makes THIS directory its own standalone git repo (independent of the parent
# WWW repo), adds every file in the tree that is not excluded by .gitignore,
# commits and pushes to GitLab.
#
# Usage: ./git_push.sh "Your commit message"
#        ./git_push.sh                 (default message with timestamp)
#
# Auth: uses Kerberos.  Run `kinit agapitos@CERN.CH` first if the push asks
# for a password.
# =============================================================================

set -e  # exit on error

# --- always operate on the directory this script lives in ---------------------
cd "$(dirname "$(readlink -f "$0")")"
REPO_DIR="$(pwd)"

# GitLab repository URL (Kerberos auth for lxplus, port 8443)
GITLAB_REPO="https://:@gitlab.cern.ch:8443/agapitos/tt1lWcb.git"
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

# remote
if ! git remote | grep -q '^origin$'; then
    git remote add origin "$GITLAB_REPO"
    echo -e "${YELLOW}Added remote origin -> ${GITLAB_REPO}${NC}"
else
    git remote set-url origin "$GITLAB_REPO"
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

echo -e "${YELLOW}Pushing to origin/${BRANCH} ...${NC}"
git push -u origin "$BRANCH" || {
    echo -e "${YELLOW}Retrying with --set-upstream...${NC}"
    git push --set-upstream origin "$BRANCH"
}

echo -e "${GREEN}=== Done! ===${NC}"
