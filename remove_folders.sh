#!/bin/bash

# Clone the repository if you haven't already
# git clone https://github.com/your-username/your-repo.git

# Set REPO_PATH to the current working directory
REPO_PATH="$(pwd)"

cd "$REPO_PATH"

# Fetch all branches from the remote
git fetch origin

# Get a list of all remote branches and iterate through them
for branch in $(git branch -r | grep origin | sed 's/^ *origin\///'); do
  # Check out the branch
  git checkout "$branch"

  # Remove all .vscode folders
  find . -type d -name ".vscode" -exec rm -r {} +

  # Check if anything has been deleted
  if [[ -n "$(git status --porcelain)" ]]; then
    # Stage, commit, and push the changes
    git add .
    git commit -m "Remove all .vscode folders"
    git push origin "$branch"
  fi
done

# Optional: Switch back to the main branch (change 'main' to 'master' if needed)
git checkout main
