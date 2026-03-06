#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building TSLF..."
make

echo "Testing TSLF with sample input..."
echo ""
echo "=== Test 1: Text summarization (50 words) ==="
printf "tests/sample_input.txt\nsummary_50.txt\n50\n" | ./tslf

echo ""
echo "=== Output Summary ==="
cat summary_50.txt

echo ""
echo "=== Test 2: Larger summary (80 words) ==="
printf "tests/sample_input.txt\nsummary_80.txt\n80\n" | ./tslf

echo ""
echo "=== Output Summary ==="
cat summary_80.txt
