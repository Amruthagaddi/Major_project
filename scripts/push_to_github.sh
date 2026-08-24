#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Amruthagaddi/Major_project.git"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Working in $ROOT_DIR"
cd "$ROOT_DIR"

echo "Setting remote origin to $REPO_URL"
git remote set-url origin "$REPO_URL"

echo "Ensuring branch is 'main'"
git branch -M main || true

echo "Fetching origin"
git fetch origin

echo "Creating backup branch 'backup-before-push'"
git branch -f backup-before-push

echo "Pulling remote main and allowing unrelated histories (safe merge)"
git pull origin main --allow-unrelated-histories --no-rebase || {
  echo "Pull reported conflicts. Resolve them, then run: git add <files>; git commit; git push -u origin main"
  exit 1
}

echo "Pushing to origin/main"
git push -u origin main

echo "Done. If push failed due to auth/network, try one of these:" 
echo "  - git remote set-url origin git@github.com:Amruthagaddi/Major_project.git && git push -u origin main" 
echo "  - git push -u --force origin main    # destructive: overwrites remote"
