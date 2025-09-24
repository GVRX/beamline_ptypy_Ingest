# --- VS Code (portable) pinned to 1.85.2 for glibc 2.17 hosts ---
VSCODE_VER="1.85.2"
PREFIX="${HOME}/portable-tools"   # or wherever you're installing
mkdir -p "$PREFIX"

echo "Downloading VS Code $VSCODE_VER..."
curl -L "https://update.code.visualstudio.com/${VSCODE_VER}/linux-x64/stable" -o "$PREFIX/vscode-${VSCODE_VER}.tar.gz"

echo "Extracting..."
tar -xzf "$PREFIX/vscode-${VSCODE_VER}.tar.gz" -C "$PREFIX"
rm "$PREFIX/vscode-${VSCODE_VER}.tar.gz"

# Make it 'portable' by adding a data dir alongside the app
mv "$PREFIX/VSCode-linux-x64" "$PREFIX/VSCode-${VSCODE_VER}-linux-x64"
mkdir -p "$PREFIX/VSCode-${VSCODE_VER}-linux-x64/data"

# Symlink a handy launcher
ln -sf "$PREFIX/VSCode-${VSCODE_VER}-linux-x64/bin/code" "$PREFIX/code-1.85"

echo "Add to your shell rc:"
echo "  export PATH=\"$PREFIX:\$PATH\""
echo "Then run 'code-1.85'."
