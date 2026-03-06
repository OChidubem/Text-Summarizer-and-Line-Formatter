# TSLF Documentation

Technical algorithm details and design notes for TSLF live in this document set.

## Core Summary Pipeline

1. Sentence splitting with abbreviation handling
2. Tokenization and stopword filtering
3. Similarity graph construction using cosine similarity
4. TextRank scoring (PageRank-style)
5. MMR sentence selection for diversity
6. Summary assembly and compression stats

## Build Docs

```bash
doxygen docs/Doxyfile
```

If LaTeX output is generated, build PDF with:

```bash
./scripts/build_pdf.sh
```
