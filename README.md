# FAQtor

Semantic FAQ retrieval system built without LLMs or external APIs. Answers are matched against a curated dataset using dense sentence embeddings, with TF-IDF as a comparative baseline.

## Tech Stack

- **Language**: Python 3.12+
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (CPU-only, cached locally)
- **Similarity**: Cosine similarity via dot product on normalized vectors
- **Baseline**: TF-IDF with unigrams and bigrams

## Key Features

- Semantic search over 156 FAQ entries with a calibrated similarity threshold (0.54)
- Persistent embedding cache validated by question hash and model revision
- Comparative evaluation report (embeddings vs. TF-IDF) with false positive/negative analysis
- CLI interface for both interactive and single-query modes

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python faq-cli.py                                              # interactive CLI
python main.py search "how do I enroll?"                       # single query
python main.py search "how do I drop a course?" --top-k 5     # with options
python main.py evaluate                                        # run comparison report
pytest                                                         # automated tests
```

> On first run, the model weights are downloaded once and cached locally by Hugging Face.
