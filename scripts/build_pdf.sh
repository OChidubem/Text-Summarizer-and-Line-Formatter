#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

doxygen docs/Doxyfile

cd "$ROOT_DIR/docs/doxygen_output/latex"
echo "Building PDF from LaTeX files..."
make pdf 2>&1 | tail -20

if [ -f refman.pdf ]; then
  echo "refman.pdf created successfully"
  cp refman.pdf "$ROOT_DIR/docs/doxygen_documentation.pdf"
  echo "Copied to docs/doxygen_documentation.pdf"
else
  echo "PDF generation failed"
  ls -la refman.* 2>&1 | tail -10
  exit 1
fi
