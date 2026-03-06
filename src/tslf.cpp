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

string trim(const string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool has_extension(const string& filename, const string& ext) {
    if (filename.size() < ext.size()) return false;
    string tail = filename.substr(filename.size() - ext.size());
    string extLower = ext;
    std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);
    return tail == extLower;
}

bool convert_pdf_to_text(const string& pdfPath, string& tempTxtPath) {
    tempTxtPath = pdfPath + ".tslf_tmp.txt";
    string cmd = "pdftotext -layout \"" + pdfPath + "\" \"" + tempTxtPath + "\"";

    int result = system(cmd.c_str());
    if (result != 0) {
        cerr << "Error: pdftotext not available. Please install with:\n"
             << "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
             << "  macOS: brew install poppler\n"
             << "  Or use a .txt file instead.\n";
        return false;
    }

    ifstream test(tempTxtPath.c_str());
    if (!test.is_open()) {
        cerr << "Error: converted text file not found: " << tempTxtPath << "\n";
        return false;
    }
    test.close();
    return true;
}

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

    if (!tempPath.empty()) {
        remove(tempPath.c_str());
    }

    return true;
}

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

vector<double> textrank(const vector<string>& sentences) {
    size_t n = sentences.size();
    if (n == 0) return {};

    vector<vector<string>> tokens(n);
    for (size_t i = 0; i < n; ++i)
        tokens[i] = tokenize(sentences[i]);

    vector<vector<double>> graph(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = cosine_similarity(tokens[i], tokens[j]);
            if (sim == 0.0) continue;
            int distance = (int)j - (int)i;
            double decay = 1.0 / (1.0 + 0.1 * distance);
            graph[i][j] = graph[j][i] = sim * decay;
        }
    }

    vector<double> scores(n, 1.0);
    const double d = 0.85;
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

    double avg = 0.0;
    for (auto s : scores) avg += s;
    avg /= n;
    if (n >= 1) scores[0] *= (1.0 + avg);
    if (n >= 2) scores[1] *= 1.3;

    return scores;
}

vector<size_t> select_sentences_textrank(const vector<string>& sentences,
                                         const vector<double>& scores,
                                         size_t target_words) {
    size_t n = sentences.size();
    vector<pair<double, size_t>> ranked;
    for (size_t i = 0; i < n; ++i)
        if (sentences[i].size() > 20)
            ranked.emplace_back(scores[i], i);

    sort(ranked.rbegin(), ranked.rend());

    vector<size_t> selected;
    vector<vector<string>> selected_tokens;
    size_t current_words = 0;
    const double lambda = 0.7;

    for (const auto& p : ranked) {
        size_t idx = p.second;
        const auto& toks = tokenize(sentences[idx]);

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

    if (selected.empty() && n > 0) {
        size_t fallback_count = min(n, size_t(3));
        for (size_t i = 0; i < fallback_count; ++i) {
            if (i < ranked.size())
                selected.push_back(ranked[i].second);
        }
    }
    if (selected.empty() && n > 0) selected.push_back(0);
    sort(selected.begin(), selected.end());
    return selected;
}

string build_summary(const vector<string>& sentences, const vector<size_t>& idx) {
    ostringstream oss;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (i > 0) oss << " ";
        oss << sentences[idx[i]];
    }
    return oss.str();
}

bool read_target_words(size_t& n) {
    cout << "How many words in the summary? ";
    if (!(cin >> n) || n == 0) return false;
    if (n > 10000) {
        cout << "Warning: capping target at 10000 words.\n";
        n = 10000;
    }
    return true;
}

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
