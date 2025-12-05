#!/bin/bash
cd /workspaces/Text-Summarizer-and-Line-Formatter/doxygen_output/latex
echo "Building PDF from LaTeX files..."
make pdf 2>&1 | tail -20
if [ -f refman.pdf ]; then
  echo "✓ refman.pdf created successfully!"
  cp refman.pdf /workspaces/Text-Summarizer-and-Line-Formatter/doxygen_documentation.pdf
  echo "✓ Copied to doxygen_documentation.pdf"
else
  echo "✗ PDF generation failed"
  ls -la refman.* 2>&1 | tail -10
fi
