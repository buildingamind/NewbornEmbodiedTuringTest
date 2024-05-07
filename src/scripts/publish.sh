#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

# Change directory to one level up from the script location and then into src/nett
cd "$SCRIPT_DIR/../nett"

# Define the version variable by reading the version from the version.py file
VERSION=$(python3 -c "from version import __version__; print(__version__)")

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
cd "$SCRIPT_DIR/../../"

echo -e "\033[1m Updating build package... \033[0m"

python3 -m pip install --upgrade build

echo -e "\033[1m Updating Twine package... \033[0m"

python3 -m pip install --upgrade twine

# Remove all files inside dist/
rm -rf dist/*

# Switch to the main branch and pull the latest changes
git checkout main
git pull origin main

echo -e "\033[1m Tagging... \033[0m"
# Tag the latest commit with the version number
git tag v$VERSION

# Push the tag to the remote repository
git push origin v$VERSION

echo -e "\033[1m Creating Release Branch... \033[0m"

# Create a new branch with the name based on the VERSION variable
git checkout -b release/$VERSION

# Push the new branch to the remote repository
git push origin release/$VERSION

echo -e "\033[1m Building Distribution Archives... \033[0m"

python3 -m build

echo -e "\033[1m Building Distribution Archives... \033[0m"

python3 -m twine upload --repository nett-benchmarks dist/*
