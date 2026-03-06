#!/bin/bash
cd /workspaces/Text-Summarizer-and-Line-Formatter
echo "Testing TSLF with sample input..."
echo ""
echo "=== Test 1: Text summarization (50 words, plain text) ==="
echo -e "sample_input.txt\nsummary_50.txt\n50\n1" | ./tslf
echo ""
echo "=== Output Summary ==="
cat summary_50.txt
echo ""
echo ""
echo "=== Test 2: Larger summary (80 words, plain text) ==="
echo -e "sample_input.txt\nsummary_80.txt\n80\n1" | ./tslf
echo ""
echo "=== Output Summary ==="
cat summary_80.txt
echo ""
echo ""
echo "=== Test 3: Bullet point summary (50 words) ==="
echo -e "sample_input.txt\nsummary_bullets.txt\n50\n2" | ./tslf
echo ""
echo "=== Output Summary ==="
cat summary_bullets.txt
