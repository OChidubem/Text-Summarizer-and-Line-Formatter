# Text-Summarizer-and-Line-Formatter

TSLF is a C++ command-line text summarizer that uses a TextRank-style graph algorithm to generate extractive summaries from `.txt` and `.pdf` inputs.

## Features

- TextRank-inspired sentence scoring
- MMR-based sentence selection to reduce redundancy
- Position-aware weighting for better coherence
- Interactive CLI workflow
- PDF support via `pdftotext` (optional)

## Requirements

- Linux/macOS with `g++`
- C++11 or newer
- Optional for PDF input: `pdftotext` from Poppler

Install PDF support on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

## Build

From the repository root:

```bash
g++ -std=c++11 -O2 -Wall tslf.cpp -o tslf
```

This creates the executable `./tslf`.

## Run

```bash
./tslf
```

You will be prompted for:

1. Input file path (`.txt` or `.pdf`)
2. Output file path
3. Target summary word count (`1` to `10000`)

Example session:

```text
Ultra-Accurate TextRank Summarizer (TXT + PDF)

Input file (.txt or .pdf): sample_input.txt
Output file: summary.txt
How many words in the summary? 50
Summary written to 'summary.txt' (46 words, 38.0% compression).
```

## Quick Test

The repo includes a simple test script:

```bash
chmod +x test_summarizer.sh
./test_summarizer.sh
```

This runs two sample summarizations and prints generated output.

## Project Files

- `tslf.cpp`: Main C++ source code
- `test_summarizer.sh`: Basic test runner for sample inputs
- `sample_input.txt`: Example input text
- `DOCUMENTATION.md`: Extended technical documentation
- `Doxyfile`: Doxygen configuration
- `doxygen_documentation.html`: Generated API docs in HTML form
- `build_pdf.sh`: Helper script to build Doxygen PDF output

## Generate Doxygen Docs

Generate documentation files:

```bash
doxygen Doxyfile
```

If LaTeX output exists and you want a PDF:

```bash
chmod +x build_pdf.sh
./build_pdf.sh
```

## Exit Codes and Errors

- Returns non-zero on invalid input, missing files, or output write failure
- If PDF conversion tools are missing, use `.txt` input or install Poppler

## License

No explicit license file is currently included in this repository.