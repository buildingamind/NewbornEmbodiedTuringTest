#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

# Change directory to one level up from the script location and then into src/nett
cd "$SCRIPT_DIR/../src/nett/"

# Define the version variable by reading the version from the version.py file
VERSION=$(python3 -c "from _version import __version__; print(__version__)")

# Verify that VERSION is retrieved successfully
if [[ -z "$VERSION" ]]; then
    echo "Failed to retrieve VERSION. Exiting..."
    exit 1
fi

# Confirm with the user
echo -e "The retrieved version is \033[1m$VERSION\033[0m. Do you want to continue? (yes/no)"
read user_confirmation
if [[ "$user_confirmation" != "yes" ]]; then
    echo "User cancelled the operation. Exiting..."
    exit 1
fi

echo -e "\033[1m Publishing version $VERSION... \033[0m."

# Return to base directory
cd "$SCRIPT_DIR/../"

echo -e "\033[1m Updating build package... \033[0m"

python3 -m pip install --upgrade build

echo -e "\033[1m Updating Twine package... \033[0m"

python3 -m pip install --upgrade twine

# Switch to the main branch and pull the latest changes
git checkout main
git pull origin main

# Remove all files inside dist/
if [ "$(ls -A dist/)" ]; then
  rm -rf dist/*
  echo "Files and folders in dist/ have been deleted."
else
  echo "dist/ is already empty."
fi

echo -e "\033[1m Tagging... \033[0m"
# Tag the latest commit with the version number
git tag v$VERSION

# Push the tag to the remote repository
git push origin v$VERSION

echo -e "\033[1m Creating Release Branch... \033[0m"

# Check if the branch exists and check it out if it does, otherwise create and push it
if git show-ref --quiet refs/heads/release/$VERSION; then
  echo "Branch release/$VERSION exists. Checking it out..."
  git checkout release/$VERSION
else
  echo "Branch release/$VERSION does not exist. Creating and pushing it..."
  git checkout -b release/$VERSION
  git push origin release/$VERSION
fi

echo -e "\033[1m Installing Dependencies... \033[0m"

pip install -r ./docs/requirements.txt

echo -e "\033[1m Building Distribution Archives... \033[0m"

python3 -m build

echo -e "\033[1m Uploading to PyPI... \033[0m"

python3 -m twine upload --repository pypi dist/*
