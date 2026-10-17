#!/usr/bin/env bash
# One-command dev setup.
#
# Works around a uv-on-macOS wart: uv marks its editable-install .pth files with the
# macOS UF_HIDDEN flag, and CPython's site.py skips hidden .pth files — so
# `import flashtopic` silently fails after `uv pip install -e .`. (The .pth itself is
# fine; it just adds src/ to the path. This never affects a normal `pip install
# flashtopic` from a wheel — only local editable dev with uv on macOS.)
#
# We install editable (needed for the `flashtopic` CLI entry point) and then drop a
# plain, non-hidden .pth we own, which keeps imports working across future installs.
#
# Usage: scripts/dev-setup.sh [extra]   # extra = mlx (default) | cuda | cpu
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA="${1:-mlx}"
uv venv --python 3.12 2>/dev/null || true
uv pip install -e ".[${EXTRA},dev]"
uv run python -m spacy download en_core_web_sm

SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
echo "$(pwd)/src" > "$SITE/flashtopic-dev.pth"          # non-hidden; we own it
chflags nohidden "$SITE"/*.pth 2>/dev/null || true       # belt-and-suspenders for uv's

uv run python -c "import flashtopic; print('dev env OK — flashtopic', flashtopic.__version__)"
