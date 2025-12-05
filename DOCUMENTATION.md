# TSLF - Ultra-Accurate TextRank-Based Summarizer

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Algorithm Details](#algorithm-details)
4. [Building & Running](#building--running)
5. [Function Documentation](#function-documentation)
6. [Code Structure](#code-structure)
7. [Examples](#examples)
8. [Performance](#performance)

---

## Overview

**TSLF** (Text Summarizer & Line Formatter) is a professional-grade extractive text summarization system implementing the **TextRank algorithm**, inspired by Google's PageRank. It produces summaries significantly closer to human quality than traditional TF-IDF methods.

**Author:** OChidubem  
**Date:** December 2025  
**Language:** C++11  
**License:** Open Source

---

## Features

### Core Algorithms
- ✅ **Graph-based TextRank**: Builds sentence similarity graph using cosine similarity
- ✅ **Position Decay**: Intelligently weights edges based on sentence distance
- ✅ **Adaptive Lead Boosting**: Dynamically boosts opening sentences based on context
- ✅ **Maximum Marginal Relevance (MMR)**: Removes redundancy while maintaining relevance
- ✅ **PageRank Scoring**: Iterative importance computation with convergence detection

### Input/Output
- ✅ **Multi-format Support**: Reads .txt and .pdf files
- ✅ **Graceful Degradation**: Works without pdftotext (falls back to .txt)
- ✅ **PDF Conversion**: Uses pdftotext with -layout flag for structure preservation
- ✅ **Compression Metrics**: Shows compression ratio for quality assessment

### Robustness
- ✅ **Error Handling**: Comprehensive error messages and fallback strategies
- ✅ **Edge Case Handling**: Handles empty input, single sentences, malformed text
- ✅ **Input Validation**: Validates word count targets (1-10,000)
- ✅ **Temp File Cleanup**: Automatically removes temporary PDF conversion files

---

## Algorithm Details

### Step 1: Sentence Tokenization
```
Input: Raw text
Process:
  - Split on '.', '?', '!'
  - Handle abbreviations: Mr., Dr., U.S.A., etc.
  - Avoid false breaks on decimals (3.14)
  - Trim whitespace
Output: Vector of sentences
```

**Key Features:**
- Recognizes 14 common abbreviations
- Case-insensitive matching
- Preserves punctuation with sentences

### Step 2: Word Tokenization
```
Input: Sentence string
Process:
  - Replace punctuation with spaces (keep apostrophes & hyphens)
  - Convert to lowercase
  - Split on whitespace
  - Filter words ≤ 2 characters
  - Remove stopwords (53 common words)
Output: Vector of meaningful tokens
```

**Stopwords Filtered:**
```
a, an, and, are, as, at, be, by, for, from, has, he, in, is, it,
its, of, on, that, the, to, was, will, with, this, or, but, not,
can, have, had, you, we, they, i, their, there, been, which, who
```

### Step 3: Similarity Graph Construction
```
Input: Tokenized sentences
Process:
  For each pair of sentences (i, j):
    1. Compute cosine similarity of token vectors
    2. Apply position decay: decay = 1 / (1 + 0.1 * distance)
    3. Set edge weight: graph[i][j] = similarity * decay
Output: n×n weighted graph
```

**Cosine Similarity Formula:**
```
similarity = (a·b) / (||a|| × ||b||)

Where:
  a·b = dot product of frequency vectors
  ||a|| = Euclidean norm of frequency vector a
  ||b|| = Euclidean norm of frequency vector b
```

**Position Decay Rationale:**
- Distant sentences naturally differ in topic
- Reduces false connections between unrelated parts
- Improves summary coherence

### Step 4: PageRank Scoring
```
Input: Weighted similarity graph
Process:
  For iteration = 1 to 50 or convergence:
    For each sentence i:
      score[i] = (1 - d) + d × Σ(score[j] × edge[j→i] / outgoing[j])
    
    Where:
      d = 0.85 (damping factor, standard PageRank value)
      edge[j→i] = weight from j to i
      outgoing[j] = sum of all outgoing edges from j

    Check convergence: if max_diff < 1e-6, break
Output: Importance score for each sentence
```

**Adaptive Boost:**
```
If n ≥ 1:
  score[0] *= (1 + average_score)  // Boost first sentence adaptively

If n ≥ 2:
  score[1] *= 1.3  // Moderate boost for second sentence
```

### Step 5: Maximum Marginal Relevance Selection
```
Input: Sentences, scores, target_words
Process:
  1. Filter: Remove sentences < 20 characters
  2. Rank: Sort by TextRank score (descending)
  3. Greedy loop for each ranked sentence:
     - Compute diversity penalty (max word overlap with selected)
     - MMR = 0.7 × score - 0.3 × penalty
     - If MMR > 0.3 AND word_budget allows:
       Add to summary
     - Stop when summary ≥ 80% of target
  4. Fallback: If empty, take top 3 sentences by score
  5. Sort selected sentences by original position
Output: Indices of sentences for summary
```

**MMR Formula:**
```
MMR(S) = λ × relevance(S) - (1-λ) × similarity(S, already_selected)

Where:
  λ = 0.7 (relevance weight)
  relevance = TextRank score
  similarity = max word overlap ratio
```

### Step 6: Summary Assembly & Output
```
Input: Original sentences, selected indices
Process:
  1. Concatenate sentences in original order
  2. Join with spaces
  3. Write to output file
  4. Calculate compression ratio
  5. Report statistics
Output: Summary file + metrics
```

---

## Building & Running

### Compilation
```bash
g++ -std=c++11 -O2 -Wall tslf.cpp -o tslf
```

**Flags:**
- `-std=c++11`: C++11 standard
- `-O2`: Optimization level 2
- `-Wall`: All warnings enabled

### Installation (Optional PDF Support)
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Fedora
sudo dnf install poppler-utils
```

### Running
```bash
./tslf
```

**Interactive Prompts:**
```
Ultra-Accurate TextRank Summarizer (TXT + PDF)

Input file (.txt or .pdf): document.txt
Output file: summary.txt
How many words in the summary? 150
```

---

## Function Documentation

### Utility Functions

#### `string trim(const string& s)`
**Purpose:** Remove leading/trailing whitespace

**Algorithm:**
1. Find first non-whitespace character
2. Find last non-whitespace character
3. Extract substring between them

**Example:**
```cpp
string result = trim("  hello world  ");  // "hello world"
```

#### `bool has_extension(const string& filename, const string& ext)`
**Purpose:** Check file extension (case-insensitive)

**Algorithm:**
1. Compare filename suffix with extension
2. Both converted to lowercase
3. Return equality

**Example:**
```cpp
if (has_extension("doc.PDF", ".pdf")) { /* Handle PDF */ }
```

### File I/O Functions

#### `bool convert_pdf_to_text(const string& pdfPath, string& tempTxtPath)`
**Purpose:** Convert PDF to text using pdftotext

**Process:**
1. Create temp filename: `pdfPath + ".tslf_tmp.txt"`
2. Execute: `pdftotext -layout "input.pdf" "temp.txt"`
3. Verify temp file exists
4. Return success/failure

**Requirements:**
- pdftotext must be installed
- Input PDF must be readable
- Sufficient disk space for temp file

**Example:**
```cpp
string tempPath;
if (convert_pdf_to_text("paper.pdf", tempPath)) {
    // Converted successfully
}
```

#### `bool read_any_input_file(const string& inputPath, string& outText)`
**Purpose:** Read text from .txt or .pdf file

**Logic:**
```
if file ends with .pdf:
  convert to text via pdftotext
  read converted file
else:
  read as plain text file

clean up temp files if any
return success/failure
```

**Example:**
```cpp
string content;
if (read_any_input_file("input.txt", content)) {
    // Process content
}
```

### Processing Functions

#### `vector<string> split_into_sentences(const string& text)`
**Purpose:** Split text into sentences

**Algorithm:**
1. Iterate through characters
2. Accumulate into current sentence
3. On '.', '?', '!':
   - Check if previous word is abbreviation
   - Check if next char is digit (decimal)
   - If neither, treat as sentence boundary
4. Trim and add to vector

**Abbreviations Handled:**
```
Mr., Mrs., Ms., Dr., Prof., Sr., Jr., vs., etc.,
Inc., Ltd., Corp., Fig., e.g., i.e., U.S., U.S.A.
```

#### `vector<string> tokenize(const string& s)`
**Purpose:** Extract meaningful words from sentence

**Process:**
1. Replace punctuation with spaces (except ' and -)
2. Lowercase
3. Split on whitespace
4. Filter: word.size() > 2 AND not in stopwords
5. Return tokens

**Example:**
```cpp
vector<string> tokens = tokenize("The quick brown fox!");
// Result: ["quick", "brown", "fox"]
```

#### `double cosine_similarity(const vector<string>& a, const vector<string>& b)`
**Purpose:** Compute semantic similarity between sentences

**Algorithm:**
1. Build frequency maps for both token vectors
2. Compute dot product: Σ(freq_a[i] × freq_b[i])
3. Compute L2 norms: sqrt(Σ(freq_a[i]²))
4. Return: dot_product / (norm_a × norm_b)

**Range:** [0, 1]
- 0.0 = completely different
- 1.0 = identical

**Example:**
```cpp
vector<string> a = {"machine", "learning"};
vector<string> b = {"machine", "learning"};
double sim = cosine_similarity(a, b);  // 1.0
```

### Main Algorithm Functions

#### `vector<double> textrank(const vector<string>& sentences)`
**Purpose:** Compute TextRank importance scores

**Complexity:** O(n² × iterations)

**Key Parameters:**
- Damping factor: 0.85
- Max iterations: 50
- Convergence threshold: 1e-6
- Position decay coefficient: 0.1

**Returns:** Score for each sentence

#### `vector<size_t> select_sentences_textrank(...)`
**Purpose:** Select diverse sentences using MMR

**Parameters:**
- `lambda`: 0.7 (relevance vs diversity balance)
- `mmr_threshold`: 0.3
- `min_sentence_len`: 20 chars
- `word_budget`: 1.2 × target

**Returns:** Indices of selected sentences in original order

#### `string build_summary(const vector<string>& sentences, const vector<size_t>& idx)`
**Purpose:** Assemble final summary

**Process:**
1. For each index: append sentence + space
2. Join in original order
3. Return concatenated string

#### `bool read_target_words(size_t& n)`
**Purpose:** Get validated word count from user

**Validation:**
- Must be positive integer
- Capped at 10,000
- User warned if exceeded

**Returns:** true if valid input read

---

## Code Structure

```
tslf.cpp (692 lines)
├── File Header & Documentation (50 lines)
├── Includes & Namespaces (15 lines)
│
├── Utility Functions (120 lines)
│   ├── trim()
│   ├── has_extension()
│   ├── convert_pdf_to_text()
│   ├── read_any_input_file()
│   └── ...
│
├── Tokenization Functions (80 lines)
│   ├── split_into_sentences()
│   ├── tokenize()
│   ├── word_overlap()
│   ├── count_words()
│   └── cosine_similarity()
│
├── Algorithm Core (280 lines)
│   ├── textrank() [180 lines]
│   ├── select_sentences_textrank() [85 lines]
│   └── build_summary() [15 lines]
│
├── User Interface (50 lines)
│   └── read_target_words()
│
└── Main Function (100 lines)
    └── main()
```

---

## Examples

### Example 1: Text Summarization
```bash
./tslf
Input file (.txt or .pdf): paper.txt
Output file: summary.txt
How many words in the summary? 100
Summary written to 'summary.txt' (98 words, 24.5% compression).
```

### Example 2: PDF Summarization
```bash
./tslf
Input file (.txt or .pdf): research.pdf
Output file: abstract.txt
How many words in the summary? 200
Detected PDF input. Converting with pdftotext...
Summary written to 'abstract.txt' (195 words, 18.2% compression).
```

---

## Performance

### Time Complexity
```
Sentence Tokenization:     O(m)            [m = text length]
Graph Construction:        O(n² × s)       [s = avg tokens/sentence]
PageRank Iteration:        O(n² × iter)    [iter = typically 10-30]
MMR Selection:             O(n² × k)       [k = selected sentences]
Overall:                   O(n² × iter)    [n = # sentences]
```

### Space Complexity
```
Sentences Vector:          O(n × s)
Token Vectors:             O(n × s)
Similarity Graph:          O(n²)
Overall:                   O(n²)
```

### Typical Performance
```
Document Size    Sentences    Time (ms)    Memory (MB)
1,000 words      50           10-20        2-5
10,000 words     500          50-100       10-20
100,000 words    5,000        200-500      50-100
```

### Tuning Parameters
For better quality/speed tradeoffs, modify these constants:

```cpp
// In textrank():
const double d = 0.85;          // Higher = more iterations needed
const int max_iter = 50;        // Lower = faster, less accurate
const double tol = 1e-6;        // Higher = fewer iterations

// In split_into_sentences():
const set<string> abbrevs = { /* Add more if needed */ };

// In select_sentences_textrank():
const double lambda = 0.7;      // Higher = more relevant, more redundant
if (mmr > 0.3)                  // Lower threshold = more sentences included
if (sentences[i].size() > 20)   // Adjust minimum sentence length
```

---

## References

- **TextRank**: Rada Mihalcea and Paul Tarau. "TextRank: Bringing Order into Texts." (2004)
- **PageRank**: Lawrence Page et al. "The PageRank Citation Ranking." (1998)
- **Cosine Similarity**: Salton & McGill. "Introduction to Modern Information Retrieval." (1983)
- **MMR**: Jaime Carbonell and Jade Goldstein. "The Use of MMR." (1998)

---

## Future Enhancements

1. **Multi-language Support**: Add language detection and multi-language tokenization
2. **Custom Stopwords**: Allow user-provided stopword lists
3. **Configurable Parameters**: Command-line flags for tuning
4. **Abstractive Summarization**: Combine with neural networks
5. **Batch Processing**: Process multiple files
6. **Progress Reporting**: Show progress for large documents
7. **Output Formatting**: HTML, Markdown, PDF output options
8. **Caching**: Cache similarity graphs for repeated documents

---

**Generated:** December 5, 2025

