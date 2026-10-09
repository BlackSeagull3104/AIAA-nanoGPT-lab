# Token Analysis

This branch organizes the token-analysis components of the nanoGPT course project.

## Scope

- Shakespeare data preparation
- Character/token frequency analysis
- Tokenizer comparison
- Embedding norm analysis

## Relevant files

- `data/shakespeare_char/prepare.py`
- `scripts/task4_tokenizer_comparison.py`
- `scripts/task6_embedding_norm.py`
- `reports/token_frequency_char.csv`
- `reports/token_frequency_char.png`
- `reports/tokenizer_comparison.csv`
- `reports/embedding_norm_statistics.json`
- `reports/embedding_norm_distribution.png`

## Reproduction

Character statistics and tokenizer analyses can be reproduced using the scripts in `scripts/`.
Generated binary datasets and model checkpoints remain excluded by `.gitignore`.

## Validation

The token analysis workflow is designed to run independently of full model training. Generated statistics should be checked for expected token counts, tokenizer behavior, and embedding dimensions.

Generated analysis outputs should remain small and suitable for version control; large binary datasets and model checkpoints should remain excluded.
