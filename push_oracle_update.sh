#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/oem/Documents/The_Oracle_TTS"
REMOTE_URL="git@github.com:kingcinder/The_Oracle_TTS.git"

echo "==> Entering repo"
cd "$REPO_DIR"

echo "==> Ensuring git repo exists"
git rev-parse --is-inside-work-tree >/dev/null

echo "==> Ensuring remote origin is correct"
if git remote | grep -q origin; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

echo "==> Staging changes"
git add -A

if git diff --cached --quiet; then
    echo "==> No changes to commit"
else
    COMMIT_MSG=${1:-"Oracle TTS update $(date +'%Y-%m-%d %H:%M:%S')"}
    echo "==> Committing: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
fi

echo "==> Pushing to GitHub"
git push origin main

echo "==> Done"
