# Text-Summarizer-and-Line-Formatter

TSLF is a C++ TextRank-based extractive summarizer for `.txt` and `.pdf` documents.
It scores sentence importance with a graph-ranking approach and applies MMR to reduce redundancy.

## Why This Project

- Demonstrates practical NLP techniques in C++ (TextRank + MMR)
- Includes PDF ingestion and robust sentence/token preprocessing
- Structured as a maintainable utility project, not a single-file dump

## Repository Layout

```text
Text-Summarizer-and-Line-Formatter/
├── src/
│   └── tslf.cpp
├── include/
├── tests/
│   ├── sample_input.txt
│   ├── expected_output.txt
│   └── fixtures/
├── docs/
│   ├── DOCUMENTATION.md
│   ├── Doxyfile
│   └── doxygen_output/
├── examples/
│   ├── before.txt
│   └── after.txt
├── scripts/
│   ├── test_summarizer.sh
│   └── build_pdf.sh
├── Makefile
├── README.md
└── .gitignore
```

## Requirements

- `g++` with C++11 support
- Optional for PDF input: `pdftotext` (Poppler)

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y g++ poppler-utils
```

## Build

```bash
make
```

or directly:

```bash
g++ -std=c++11 -O2 -Wall src/tslf.cpp -o tslf
```

## Run

```bash
./tslf
```

Prompts:

1. Input file path (`.txt` or `.pdf`)
2. Output file path
3. Target summary word count (`1` to `10000`)

Example:

```text
Ultra-Accurate TextRank Summarizer (TXT + PDF)

Input file (.txt or .pdf): tests/sample_input.txt
Output file: summary.txt
How many words in the summary? 50
Summary written to 'summary.txt' (46 words, 38.0% compression).
```

## Test

```bash
make test
```

or:

```bash
bash scripts/test_summarizer.sh
```

## Documentation

Generate API docs:

```bash
make docs
```

Build documentation PDF (if LaTeX toolchain is available):

```bash
bash scripts/build_pdf.sh
```

## Notes

- Generated binaries and temporary outputs are ignored by `.gitignore`
- Large/random fixtures are kept under `tests/fixtures/` to keep root clean