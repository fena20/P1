#!/bin/bash

# git-pr-check.sh
# A utility script to diagnose common issues before creating a pull request
# Helps prevent the "There isn't anything to compare" error on GitHub

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default base branch
BASE_BRANCH="${1:-main}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Git Pull Request Readiness Checker                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}✗ Error: Not inside a Git repository${NC}"
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
REPO_ROOT=$(git rev-parse --show-toplevel)

echo -e "${BLUE}Repository:${NC} $REPO_ROOT"
echo -e "${BLUE}Current Branch:${NC} $CURRENT_BRANCH"
echo -e "${BLUE}Base Branch:${NC} $BASE_BRANCH"
echo ""

# Track issues
ISSUES=0

# 1. Check if base branch exists
echo -e "${YELLOW}[1/5] Checking if base branch exists...${NC}"
if git show-ref --verify --quiet "refs/heads/$BASE_BRANCH"; then
    echo -e "${GREEN}  ✓ Base branch '$BASE_BRANCH' exists locally${NC}"
else
    echo -e "${RED}  ✗ Base branch '$BASE_BRANCH' does not exist locally${NC}"
    echo -e "    Try: git fetch origin $BASE_BRANCH"
    ((ISSUES++))
fi

# 2. Check if current branch is pushed to remote
echo ""
echo -e "${YELLOW}[2/5] Checking if current branch exists on remote...${NC}"
if git ls-remote --heads origin "$CURRENT_BRANCH" | grep -q "$CURRENT_BRANCH"; then
    echo -e "${GREEN}  ✓ Branch '$CURRENT_BRANCH' exists on remote${NC}"
    
    # Check if local is ahead of remote
    LOCAL_COMMIT=$(git rev-parse HEAD)
    REMOTE_COMMIT=$(git rev-parse "origin/$CURRENT_BRANCH" 2>/dev/null || echo "none")
    
    if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
        UNPUSHED=$(git rev-list "origin/$CURRENT_BRANCH..HEAD" --count 2>/dev/null || echo "unknown")
        echo -e "${YELLOW}  ⚠ You have $UNPUSHED unpushed commit(s)${NC}"
        echo -e "    Run: git push origin $CURRENT_BRANCH"
        ((ISSUES++))
    fi
else
    echo -e "${RED}  ✗ Branch '$CURRENT_BRANCH' does NOT exist on remote${NC}"
    echo -e "    Run: git push -u origin $CURRENT_BRANCH"
    ((ISSUES++))
fi

# 3. Check for commits ahead of base branch
echo ""
echo -e "${YELLOW}[3/5] Checking for new commits compared to '$BASE_BRANCH'...${NC}"
COMMITS_AHEAD=$(git rev-list "$BASE_BRANCH..HEAD" --count 2>/dev/null || echo "0")

if [ "$COMMITS_AHEAD" -gt 0 ]; then
    echo -e "${GREEN}  ✓ You have $COMMITS_AHEAD commit(s) ahead of '$BASE_BRANCH'${NC}"
    echo ""
    echo -e "    ${BLUE}Commits to be included in PR:${NC}"
    git log "$BASE_BRANCH..HEAD" --oneline | while read -r line; do
        echo -e "      • $line"
    done
else
    echo -e "${RED}  ✗ No new commits compared to '$BASE_BRANCH'${NC}"
    echo -e "    Your branch is identical to '$BASE_BRANCH' - nothing to compare!"
    echo -e "    Make some changes and commit them first."
    ((ISSUES++))
fi

# 4. Check for uncommitted changes
echo ""
echo -e "${YELLOW}[4/5] Checking for uncommitted changes...${NC}"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}  ⚠ You have uncommitted changes:${NC}"
    git status --short | while read -r line; do
        echo -e "      $line"
    done
    echo -e "    Consider committing these before creating a PR."
else
    echo -e "${GREEN}  ✓ Working tree is clean${NC}"
fi

# 5. Check if branch has been merged
echo ""
echo -e "${YELLOW}[5/5] Checking if branch has already been merged...${NC}"
if git branch --merged "$BASE_BRANCH" | grep -q "^\*\? *$CURRENT_BRANCH$"; then
    echo -e "${RED}  ✗ Branch '$CURRENT_BRANCH' has already been merged into '$BASE_BRANCH'${NC}"
    echo -e "    This branch contains no unique commits."
    ((ISSUES++))
else
    echo -e "${GREEN}  ✓ Branch has not been merged yet${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your branch is ready for a pull request.${NC}"
    echo ""
    echo -e "  Create a PR with:"
    echo -e "    gh pr create --base $BASE_BRANCH"
else
    echo -e "${RED}✗ Found $ISSUES issue(s) that may prevent creating a PR.${NC}"
    echo -e "  Please address the issues above before creating a pull request."
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

exit $ISSUES
