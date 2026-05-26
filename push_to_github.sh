#!/usr/bin/env bash
# Create a public-but-unlisted GitHub repo and push the current folder.
# "Unlisted" effect = no description, no topics, minimal README.
# Anyone with the URL can read; nothing surfaces in GitHub search.
#
# Prereqs:
#   - gh CLI installed and authenticated  (run: gh auth status)
#   - git installed
#
# Usage:
#   bash push_to_github.sh [REPO_NAME]
#   (default REPO_NAME is "analog-stress-test")

set -e
REPO_NAME="${1:-analog-stress-test}"

if [ ! -d .git ]; then
  git init -q
  git branch -M main
fi

git add -A
git diff --cached --quiet || git commit -q -m "Initial commit"

gh repo create "$REPO_NAME" \
    --public \
    --description "" \
    --disable-issues \
    --disable-wiki \
    --source=. \
    --remote=origin \
    --push

REMOTE_URL=$(gh repo view "$REPO_NAME" --json url -q .url)
echo
echo "Done. Share this link only:"
echo "  $REMOTE_URL"
