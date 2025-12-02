# Git PR Readiness Checker

A utility script to diagnose common issues before creating a GitHub pull request. This tool helps prevent the frustrating **"There isn't anything to compare"** error.

## The Problem

When creating a pull request on GitHub, you might encounter this error:

> "There isn't anything to compare. We couldn't figure out how to compare these references, do they point to valid commits?"

This typically happens when:
- Your branch hasn't been pushed to GitHub yet
- Your branch has no new commits compared to the base branch
- Your branch has already been merged

## Installation

```bash
# Clone the repository
git clone https://github.com/fena20/P1.git
cd P1

# Make the script executable
chmod +x git-pr-check.sh

# Optionally, add to your PATH for global access
sudo ln -s "$(pwd)/git-pr-check.sh" /usr/local/bin/git-pr-check
```

## Usage

Run the script from any Git repository:

```bash
# Check against 'main' branch (default)
./git-pr-check.sh

# Check against a different base branch
./git-pr-check.sh develop
```

## What It Checks

The script performs 5 comprehensive checks:

| Check | Description |
|-------|-------------|
| **Base Branch Exists** | Verifies the target branch exists locally |
| **Remote Branch Status** | Confirms your branch is pushed to GitHub |
| **New Commits** | Counts commits ahead of the base branch |
| **Uncommitted Changes** | Warns about changes not yet committed |
| **Merge Status** | Detects if branch was already merged |

## Example Output

```
╔════════════════════════════════════════════════════════════╗
║          Git Pull Request Readiness Checker                ║
╚════════════════════════════════════════════════════════════╝

Repository: /path/to/your/repo
Current Branch: feature/my-feature
Base Branch: main

[1/5] Checking if base branch exists...
  ✓ Base branch 'main' exists locally

[2/5] Checking if current branch exists on remote...
  ✓ Branch 'feature/my-feature' exists on remote

[3/5] Checking for new commits compared to 'main'...
  ✓ You have 3 commit(s) ahead of 'main'

    Commits to be included in PR:
      • abc1234 Add new feature
      • def5678 Update tests
      • ghi9012 Fix typo

[4/5] Checking for uncommitted changes...
  ✓ Working tree is clean

[5/5] Checking if branch has already been merged...
  ✓ Branch has not been merged yet

════════════════════════════════════════════════════════════
✓ All checks passed! Your branch is ready for a pull request.

  Create a PR with:
    gh pr create --base main
════════════════════════════════════════════════════════════
```

## Exit Codes

- `0` - All checks passed, ready to create PR
- `N` - Number of issues found that need attention

## License

MIT License - feel free to use and modify as needed.
