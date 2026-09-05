#!/bin/bash
# publish.sh - One-command publish to PyPI
# Usage: bash publish.sh PASTE_YOUR_TOKEN_HERE
# Or set PYPI_TOKEN environment variable first

set -e

cd "$(dirname "$0")"

TOKEN="${1:-$PYPI_TOKEN}"
if [ -z "$TOKEN" ]; then
    echo "Error: No PyPI token provided."
    echo "Usage: PYPI_TOKEN=pypi-xxx bash publish.sh"
    echo "   or: bash publish.sh pypi-xxx"
    exit 1
fi

echo "=== Building agent-safety-bench ==="
python -m build
echo ""
echo "=== Uploading to PyPI ==="
python -m twine upload dist/* --username __token__ --password "$TOKEN"
echo ""
echo "=== Done! ==="
echo "https://pypi.org/project/agent-safety-bench/"