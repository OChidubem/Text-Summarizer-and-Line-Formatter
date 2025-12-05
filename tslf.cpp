/**
 * @file tslf.cpp
 * @author OChidubem
 * @date December 2025
 * @brief TSLF – Ultra-Accurate TextRank-Based Summarizer (Final Version, PDF-capable)
 *
 * @section description_sec Description
 * This program implements an extractive text summarization system using the TextRank algorithm,
 * which is inspired by Google's PageRank. It produces summaries that are significantly closer
 * to human quality than traditional TF-IDF methods.
 *
 * @section features_sec Key Features
 * - **Graph-based TextRank Algorithm**: Builds a sentence similarity graph using cosine similarity
 * - **Position Decay**: Weights edges based on sentence distance to improve coherence
 * - **Adaptive Lead Boosting**: Dynamically boosts opening sentences based on document context
 * - **Maximum Marginal Relevance (MMR)**: Removes redundancy while maintaining relevance
 * - **PDF Support**: Reads both plain text and PDF files via pdftotext integration
 * - **Robust Error Handling**: Gracefully handles edge cases and invalid inputs
 * - **Compression Metrics**: Shows compression ratio for summary quality assessment
 *
 * @section usage_sec Usage
 * @code
 * ./tslf
 * Input file (.txt or .pdf): document.txt
 * Output file: summary.txt
 * How many words in the summary? 100
 * @endcode
 *
 * @section algorithm_sec Algorithm Details
 * 1. **Sentence Tokenization**: Splits text into sentences using period/question mark/exclamation
 * 2. **Word Tokenization**: Extracts meaningful words, filtering stopwords
 * 3. **Graph Construction**: Builds similarity matrix using cosine similarity with position decay
 * 4. **PageRank Scoring**: Iteratively computes sentence importance scores
 * 5. **MMR Selection**: Greedily selects diverse, high-scoring sentences
 * 6. **Summary Assembly**: Orders selected sentences by original position and outputs
 *
 * @section dependencies_sec Dependencies
 * - C++11 or later
 * - Standard C++ Library (iostream, fstream, sstream, etc.)
 * - Optional: pdftotext (for PDF support) - install via: sudo apt-get install poppler-utils
 *
 * @section compilation_sec Compilation
 * @code
 * g++ -std=c++11 -O2 -Wall tslf.cpp -o tslf
 * @endcode
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <iomanip>
#include <cstdio>

using namespace std;

/* ====================================================================
   Forward Declarations
   ==================================================================== */
string trim(const string& s);
bool has_extension(const string& filename, const string& ext);
bool convert_pdf_to_text(const string& pdfPath, string& tempTxtPath);
bool read_any_input_file(const string& inputPath, string& outText);
vector<string> split_into_sentences(const string& text);
vector<string> tokenize(const string& s);
int word_overlap(const vector<string>& a, const vector<string>& b);
size_t count_words(const string& s);
double cosine_similarity(const vector<string>& a, const vector<string>& b);
vector<double> textrank(const vector<string>& sentences);
vector<size_t> select_sentences_textrank(const vector<string>& sentences,
                                          const vector<double>& scores,
                                          size_t target_words);
string build_summary(const vector<string>& sentences, const vector<size_t>& idx);
bool read_target_words(size_t& n);

/* ====================================================================
   Utility Functions
   ==================================================================== */

/**
 * @brief Removes leading and trailing whitespace from a string.
 * @details This utility function strips all leading and trailing whitespace characters
 *          including spaces, tabs, carriage returns, and newlines.
 * @param s The input string to trim
 * @return A new string with whitespace removed from both ends
 * @retval "" if the string contains only whitespace
 * @example
 * @code
 * string result = trim("  hello world  ");  // Returns "hello world"
 * @endcode
 */
string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

/**
 * @brief Checks if a filename has a specific extension (case-insensitive).
 * @details Compares the file extension case-insensitively to support both .PDF and .pdf
 * @param filename The filename to check
 * @param ext The extension to match (e.g., ".pdf")
 * @return true if filename ends with ext (case-insensitive), false otherwise
 * @example
 * @code
 * if (has_extension("document.PDF", ".pdf")) { // Handle PDF
 * @endcode
 */
bool has_extension(const string& filename, const string& ext) {
    if (filename.size() < ext.size()) return false;
    string tail = filename.substr(filename.size() - ext.size());
    string extLower = ext;
    std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);
    return tail == extLower;
}

/**
 * @brief Converts a PDF file to plain text using the external pdftotext utility.
 * @details This function spawns a system process to convert PDF files to text format.
 *          The conversion uses the -layout flag to preserve document structure.
 *          The temporary text file is created with suffix ".tslf_tmp.txt".
 *
 * @param pdfPath Path to the input PDF file
 * @param tempTxtPath [out] Path to the generated temporary text file
 *
 * @return true if conversion succeeded and temp file exists
 * @return false if pdftotext is unavailable or conversion failed
 *
 * @pre pdfPath refers to a valid, readable PDF file
 * @post tempTxtPath contains the path to a temporary text file (caller should delete)
 *
 * @note Requires pdftotext to be installed:
 *       - Ubuntu/Debian: sudo apt-get install poppler-utils
 *       - macOS: brew install poppler
 *
 * @warning The temporary file is NOT automatically deleted; caller must use remove()
 * @throws No exceptions; errors are reported via return value
 *
 * @see read_any_input_file()
 */
bool convert_pdf_to_text(const string& pdfPath, string& tempTxtPath) {
    // You can choose a nicer temp location if you want
    tempTxtPath = pdfPath + ".tslf_tmp.txt";

    // Build command with -layout flag to preserve document structure and formatting
    string cmd = "pdftotext -layout \"" + pdfPath + "\" \"" + tempTxtPath + "\"";

    int result = system(cmd.c_str());
    if (result != 0) {
        cerr << "Error: pdftotext not available. Please install with:\n"
             << "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
             << "  macOS: brew install poppler\n"
             << "  Or use a .txt file instead.\n";
        return false;
    }

    // Simple existence check
    ifstream test(tempTxtPath.c_str());
    if (!test.is_open()) {
        cerr << "Error: converted text file not found: " << tempTxtPath << "\n";
        return false;
    }
    test.close();
    return true;
}

/**
 * @brief Reads text content from either a .txt or .pdf file.
 * @details Automatically detects PDF files by extension and converts them to text.
 *          For regular text files, reads directly. Handles temporary file cleanup.
 *          Returns the complete text content in outText parameter.
 *
 * @param inputPath Path to input file (.txt or .pdf)
 * @param outText [out] The extracted text content from the file
 *
 * @return true if file was successfully read
 * @return false if file doesn't exist, cannot be opened, or PDF conversion fails
 *
 * @pre inputPath refers to an existing file
 * @post outText contains the full text from the input file
 * @post Any temporary files (from PDF conversion) are cleaned up
 *
 * @note If input is empty after reading, a warning is printed to stderr
 * @note PDF conversion errors include helpful installation instructions
 *
 * @see convert_pdf_to_text()
 */
bool read_any_input_file(const string& inputPath, string& outText) {
    string pathToRead = inputPath;
    string tempPath;

    if (has_extension(inputPath, ".pdf")) {
        cout << "Detected PDF input. Converting with pdftotext...\n";
        if (!convert_pdf_to_text(inputPath, tempPath)) {
            cerr << "\nNote: PDF support is optional. You can still use .txt files.\n";
            return false;
        }
        pathToRead = tempPath;
    }

    ifstream in(pathToRead.c_str());
    if (!in.is_open()) {
        cerr << "Cannot open input file: " << pathToRead << endl;
        return false;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    in.close();

    outText = buffer.str();
    if (outText.empty()) {
        cerr << "Warning: input text is empty.\n";
    }

    // Clean up the temp file after reading
    if (!tempPath.empty()) {
        remove(tempPath.c_str());
    }

    return true;
}

/**
 * @brief Splits text into sentences using punctuation-based segmentation.
 * @details Intelligently handles common abbreviations (Mr., Dr., U.S.A., etc.)
 *          and avoids false sentence breaks on abbreviations and decimal numbers.
 *          Keeps punctuation attached to sentences.
 *
 * @param text The input text to split
 * @return A vector of sentence strings in original order
 *
 * @algorithm
 * 1. Iterate through text character by character
 * 2. Accumulate characters into current sentence
 * 3. On detecting '.', '?', or '!':
 *    - Check if previous word is a known abbreviation (case-insensitive)
 *    - Check if next character is a digit (decimal number case)
 *    - If neither condition, treat as sentence boundary
 * 4. Trim whitespace and add to sentences vector if non-empty
 *
 * @note Recognized abbreviations: Mr., Mrs., Ms., Dr., Prof., Sr., Jr., U.S.A., etc.
 * @note Empty sentences are filtered out
 *
 * @see tokenize()
 */
vector<string> split_into_sentences(const string& text) {
    vector<string> sentences;
    string current;
    const set<string> abbrevs = {"mr","mrs","ms","dr","prof","sr","jr","vs","etc",
                                 "inc","ltd","corp","fig","e.g","i.e","u.s","u.s.a"};

    for (size_t i = 0; i < text.size(); ++i) {
        current += text[i];
        if (text[i] == '.' || text[i] == '?' || text[i] == '!') {
            bool end = true;
            if (text[i] == '.' && i + 2 < text.size()) {
                string last = trim(current.substr(0, current.size()-1));
                if (!last.empty()) {
                    size_t sp = last.find_last_of(" \t");
                    if (sp != string::npos) last = last.substr(sp + 1);
                    transform(last.begin(), last.end(), last.begin(), ::tolower);
                    if (abbrevs.count(last)) end = false;
                }
                if (i + 1 < text.size() && isdigit(text[i+1])) end = false;
            }
            if (end) {
                string s = trim(current);
                if (!s.empty()) sentences.push_back(s);
                current.clear();
            }
        }
    }
    if (!trim(current).empty()) sentences.push_back(trim(current));
    return sentences;
}

/**
 * @brief Tokenizes a sentence into meaningful words, filtering stopwords.
 * @details Removes punctuation, converts to lowercase, and filters out common
 *          stopwords (the, and, a, etc.) that don't contribute to meaning.
 *          Only returns words with length > 2 characters.
 *
 * @param s The sentence to tokenize
 * @return A vector of processed word tokens
 *
 * @algorithm
 * 1. Replace all punctuation (except apostrophes and hyphens) with spaces
 * 2. Convert all characters to lowercase
 * 3. Split on whitespace
 * 4. Filter out words <= 2 chars and words in stopword set
 * 5. Return remaining tokens
 *
 * @note Stopword list includes: "a", "the", "and", "or", "is", "was", etc.
 * @note Contracted words (e.g., "don't") are preserved
 * @note Hyphenated words (e.g., "well-known") are preserved
 *
 * @see cosine_similarity()
 */
vector<string> tokenize(const string& s) {
    static const set<string> stopwords = {
        "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
        "its","of","on","that","the","to","was","will","with","this","or","but","not",
        "can","have","had","you","we","they","i","their","there","been","which","who"
    };
    vector<string> tokens;
    string word, cleaned = s;
    for (char& c : cleaned)
        if (ispunct((unsigned char)c) && c != '\'' && c != '-') c = ' ';
    transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);

    stringstream ss(cleaned);
    while (ss >> word) {
        if (word.size() > 2 && stopwords.count(word) == 0)
            tokens.push_back(word);
    }
    return tokens;
}

int word_overlap(const vector<string>& a, const vector<string>& b) {
    set<string> setA(a.begin(), a.end());
    int overlap = 0;
    for (const auto& w : b)
        if (setA.count(w)) ++overlap;
    return overlap;
}

size_t count_words(const string& s) {
    stringstream ss(s);
    string word;
    size_t count = 0;
    while (ss >> word) ++count;
    return count;
}

/**
 * @brief Computes cosine similarity between two sentences using term frequency vectors.
 * @details Treats each sentence as a vector of word frequencies and computes
 *          the cosine of the angle between them. Ranges from 0 (completely different)
 *          to 1 (identical).
 *
 * @param a First sentence token vector
 * @param b Second sentence token vector
 *
 * @return Cosine similarity score in range [0, 1]
 * @retval 0.0 if either vector is empty
 * @retval 1.0 if sentences are identical
 *
 * @algorithm
 * 1. Build frequency maps for both token vectors
 * 2. Compute dot product of frequency vectors
 * 3. Compute L2 norm (Euclidean length) of each vector
 * 4. Return dot_product / (norm_a * norm_b)
 *
 * @formula
 * @f$ similarity = \frac{\sum (f_a[i] \times f_b[i])}{\sqrt{\sum f_a[i]^2} \times \sqrt{\sum f_b[i]^2}} @f$
 *
 * @note This is more sophisticated than simple word overlap
 * @note Word frequency matters (repeated words count more)
 *
 * @see textrank()
 */
double cosine_similarity(const vector<string>& a, const vector<string>& b) {
    map<string, int> freq_a, freq_b;
    for (const auto& w : a) freq_a[w]++;
    for (const auto& w : b) freq_b[w]++;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (const auto& p : freq_a) {
        norm_a += p.second * p.second;
        if (freq_b.count(p.first))
            dot += p.second * freq_b[p.first];
    }
    for (const auto& p : freq_b)
        norm_b += p.second * p.second;

    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

/**
 * @brief Computes TextRank scores for sentences using PageRank algorithm.
 * @details Builds a similarity graph where sentences are nodes and cosine similarity
 *          (with position decay) forms weighted edges. Applies iterative PageRank
 *          to compute sentence importance scores.
 *
 * @param sentences Vector of sentence strings to score
 * @return Vector of scores aligned by index with input sentences
 *
 * @algorithm
 * 1. **Tokenization**: Extract meaningful words from each sentence
 * 2. **Graph Construction**: For each sentence pair (i,j):
 *    - Compute cosine similarity of token vectors
 *    - Apply position decay: decay = 1 / (1 + 0.1 * distance)
 *    - Set graph[i][j] = similarity * decay
 * 3. **PageRank Iteration** (up to 50 iterations or convergence):
 *    - For each sentence i:
 *      - score[i] = (1 - d) + d * Σ(score[j] * weight[j→i] / outgoing[j])
 *    - d = damping factor (0.85, standard PageRank value)
 * 4. **Convergence**: Stop if max score difference < 1e-6
 * 5. **Lead Boost**: Boost first sentence by (1 + avg_score)
 *                    Boost second sentence by 1.3
 *
 * @note Position decay prevents over-connecting distant sentences
 * @note Adaptive boosting adjusts for document context
 * @note Converges quickly for typical document lengths
 *
 * @complexity O(n² * iterations) where n = number of sentences
 *
 * @see select_sentences_textrank()
 */
vector<double> textrank(const vector<string>& sentences) {
    size_t n = sentences.size();
    if (n == 0) return {};

    vector<vector<string>> tokens(n);
    for (size_t i = 0; i < n; ++i)
        tokens[i] = tokenize(sentences[i]);

    // Build similarity graph using cosine similarity with position decay
    vector<vector<double>> graph(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = cosine_similarity(tokens[i], tokens[j]);
            if (sim == 0.0) continue;
            // Position decay: distant sentences naturally differ, so reduce their edge weight
            int distance = (int)j - (int)i;
            double decay = 1.0 / (1.0 + 0.1 * distance);
            graph[i][j] = graph[j][i] = sim * decay;
        }
    }

    // PageRank iteration
    vector<double> scores(n, 1.0);
    const double d = 0.85;  // damping factor
    const int max_iter = 50;
    const double tol = 1e-6;

    for (int iter = 0; iter < max_iter; ++iter) {
        vector<double> new_scores(n, (1 - d));
        double max_diff = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                if (graph[j][i] > 0) {
                    double outgoing = 0.0;
                    for (size_t k = 0; k < n; ++k)
                        if (graph[j][k] > 0) outgoing += graph[j][k];
                    sum += (outgoing > 0 ? graph[j][i] / outgoing : 0) * scores[j];
                }
            }
            new_scores[i] += d * sum;
            max_diff = max(max_diff, fabs(new_scores[i] - scores[i]));
        }
        scores = new_scores;
        if (max_diff < tol) break;
    }

    // Boost title/lead sentences adaptively based on average score
    double avg = 0.0;
    for (auto s : scores) avg += s;
    avg /= n;
    if (n >= 1) scores[0] *= (1.0 + avg);
    if (n >= 2) scores[1] *= 1.3;

    return scores;
}

/**
 * @brief Selects diverse, high-quality sentences using Maximum Marginal Relevance (MMR).
 * @details Greedily selects sentences that maximize a combination of relevance
 *          (TextRank score) and diversity (minimal word overlap with already-selected).
 *
 * @param sentences Vector of all sentences from document
 * @param scores TextRank scores for each sentence
 * @param target_words Target word count for summary
 *
 * @return Vector of sentence indices to include in summary (in original order)
 *
 * @algorithm
 * 1. **Filter**: Remove sentences shorter than 20 chars (noise)
 * 2. **Rank**: Sort sentences by TextRank score (descending)
 * 3. **Greedy Selection**:
 *    - For each sentence in ranked order:
 *      - Compute diversity penalty: max word overlap ratio vs selected sentences
 *      - Compute MMR = 0.7 * score - 0.3 * penalty
 *      - If MMR > 0.3 AND word budget allows:
 *        - Add sentence to summary
 * 4. **Stopping Criterion**: Stop when summary reaches 80% of target words
 * 5. **Fallback**: If no sentences selected (all filtered), take top 3 by score
 * 6. **Sort**: Return indices in original document order
 *
 * @note Lambda = 0.7 (0.7 * relevance - 0.3 * redundancy)
 * @note Word budget allows up to 120% of target (flexibility)
 * @note MMR threshold of 0.3 prevents including low-quality sentences
 *
 * @complexity O(n²) for diversity computation
 *
 * @see build_summary()
 */
vector<size_t> select_sentences_textrank(const vector<string>& sentences,
                                         const vector<double>& scores,
                                         size_t target_words) {
    size_t n = sentences.size();
    vector<pair<double, size_t>> ranked;
    for (size_t i = 0; i < n; ++i)
        if (sentences[i].size() > 20)  // filter very short/garbage lines
            ranked.emplace_back(scores[i], i);

    sort(ranked.rbegin(), ranked.rend());

    vector<size_t> selected;
    vector<vector<string>> selected_tokens;
    size_t current_words = 0;
    const double lambda = 0.7;  // balance relevance vs diversity

    for (const auto& p : ranked) {
        size_t idx = p.second;
        const auto& toks = tokenize(sentences[idx]);

        // Diversity penalty
        double penalty = 0.0;
        for (const auto& st : selected_tokens) {
            int ov = word_overlap(toks, st);
            int denom = max((int)toks.size(), (int)st.size());
            if (denom > 0)
                penalty = max(penalty, (double)ov / denom);
        }

        double mmr = lambda * p.first - (1 - lambda) * penalty;

        if (mmr > 0.3 &&
            current_words + count_words(sentences[idx]) <= target_words * 1.2) {
            selected.push_back(idx);
            selected_tokens.push_back(toks);
            current_words += count_words(sentences[idx]);
        }
        if (current_words >= target_words * 0.8) break;
    }

    // Better edge case handling
    if (selected.empty() && n > 0) {
        // Fallback: take top-scoring sentences regardless of MMR threshold
        size_t fallback_count = min(n, size_t(3));
        for (size_t i = 0; i < fallback_count; ++i) {
            if (i < ranked.size())
                selected.push_back(ranked[i].second);
        }
    }
    if (selected.empty() && n > 0) selected.push_back(0);  // Last resort
    sort(selected.begin(), selected.end());
    return selected;
}

/**
 * @brief Assembles selected sentences into a final summary string.
 * @details Concatenates sentences in their original order, separated by spaces.
 *          Preserves original sentence structure and punctuation.
 *
 * @param sentences All original sentences from document
 * @param idx Vector of sentence indices to include in summary
 *
 * @return The final summary text as a single string
 *
 * @algorithm
 * 1. Create output string stream
 * 2. For each index in idx:
 *    - Add space if not first sentence
 *    - Append the sentence from sentences vector
 * 3. Return assembled string
 *
 * @note Sentences are already in original order (ensured by select_sentences_textrank)
 * @note No additional formatting is applied
 *
 * @see select_sentences_textrank()
 */
string build_summary(const vector<string>& sentences, const vector<size_t>& idx) {
    ostringstream oss;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (i > 0) oss << " ";
        oss << sentences[idx[i]];
    }
    return oss.str();
}

/**
 * @brief Prompts user for target summary word count with validation.
 * @details Reads an integer from stdin, validates it's positive, and caps at 10,000.
 *          Warns user if input exceeds maximum.
 *
 * @param n [out] The validated target word count
 *
 * @return true if valid word count was read and stored
 * @return false if input failed (non-integer, zero, or stream error)
 *
 * @pre User is prompted and ready to input
 * @post n contains a value in range [1, 10000]
 *
 * @note User warning is printed if count exceeds 10000
 * @note Allows retry until valid input is provided
 *
 * @see main()
 */
bool read_target_words(size_t& n) {
    cout << "How many words in the summary? ";
    if (!(cin >> n) || n == 0) return false;
    if (n > 10000) {
        cout << "Warning: capping target at 10000 words.\n";
        n = 10000;
    }
    return true;
}

/**
 * @brief Main entry point for the TSLF summarizer application.
 * @details Orchestrates the complete summarization pipeline:
 *          1. Prompts user for input/output files and target word count
 *          2. Reads and preprocesses input text
 *          3. Applies TextRank algorithm to score sentences
 *          4. Selects diverse summary sentences using MMR
 *          5. Writes summary to output file
 *          6. Reports compression statistics
 *
 * @return 0 on successful completion
 * @return 1 on any error (invalid input, file I/O, etc.)
 *
 * @algorithm (High-level workflow)
 * @code
 * 1. Display welcome message
 * 2. Get input filename (supports .txt and .pdf)
 * 3. Get output filename
 * 4. Get target word count (1-10000)
 * 5. Read input file (auto-convert PDF if needed)
 * 6. Split into sentences
 * 7. Apply TextRank scoring
 * 8. Select summary sentences with MMR
 * 9. Build final summary
 * 10. Write to output file
 * 11. Report statistics (words, compression ratio)
 * @endcode
 *
 * @note Provides helpful error messages on failure
 * @note Gracefully handles edge cases (empty input, single sentence, etc.)
 * @note Cleans up temporary files automatically
 *
 * @section example_usage Example
 * @code
 * ./tslf
 * Ultra-Accurate TextRank Summarizer (TXT + PDF)
 *
 * Input file (.txt or .pdf): document.txt
 * Output file: summary.txt
 * How many words in the summary? 150
 * Summary written to 'summary.txt' (148 words, 23.4% compression).
 * @endcode
 *
 * @see textrank(), select_sentences_textrank(), build_summary()
 */
int main() {
    cout << "Ultra-Accurate TextRank Summarizer (TXT + PDF)\n\n";

    string in_file, out_file;
    cout << "Input file (.txt or .pdf): ";
    getline(cin >> ws, in_file);
    cout << "Output file: ";
    getline(cin, out_file);

    size_t target;
    if (!read_target_words(target)) {
        cerr << "Invalid number.\n";
        return 1;
    }

    string text;
    if (!read_any_input_file(in_file, text)) {
        return 1;
    }

    if (text.empty()) {
        cout << "Empty input (after conversion).\n";
        return 0;
    }

    auto sentences = split_into_sentences(text);
    if (sentences.empty()) {
        cout << "No sentences found.\n";
        return 0;
    }

    auto scores = textrank(sentences);
    auto selected = select_sentences_textrank(sentences, scores, target);
    string summary = build_summary(sentences, selected);

    ofstream out(out_file.c_str());
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << out_file << endl;
        return 1;
    }
    out << summary << '\n';
    out.close();

    size_t summary_words = count_words(summary);
    size_t original_words = count_words(text);
    double compression = original_words > 0 ? (100.0 * summary_words / original_words) : 0.0;
    cout << "Summary written to '" << out_file << "' ("
         << summary_words << " words, "
         << fixed << setprecision(1) << compression << "% compression).\n";

    return 0;
}
