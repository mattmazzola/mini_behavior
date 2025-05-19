#! /bin/bash

echo "CONFIGURING GIT"
git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

echo "Install Bore"
cargo install bore-cli

echo "Print Versions of CLI tools"
az version
azcopy --version
cargo --version
rustc --version
ngrok --version
uv --version
pwsh --version
dotnet --version
gh --version

echo "Install Packages"
uv sync
uv pip install -e .

echo "postCreateCommand.sh finished!"
