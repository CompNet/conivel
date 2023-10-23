# Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset


# Installing Dependencies

Use `poetry install` to install dependencies, and `poetry shell` to activate the resulting python virtual env. 


# Generated Dataset

The generated context retrieval can be found under `runs/gen/genv3` for Alpaca-7b, and under `runs/gen/genv3_13b/` for Alpaca-13b. We release the dataset under CC BY-NC 4.0 license.



# Reproducing Results

## No Retrieval Baseline

```sh
python xp_bare.py with runs/gen/gen_base_models/config.json
```

For the following experiments, we suppose you save your runs and the produced models as follows:

```sh
# supposing that the xp_bare.py run ID is 1
mv runs/1 runs/bare

# suppose that you named the models saved by xp_bare.py as follows
BARE_XP_PATH="./runs/bare"
NER_MODEL_PATHS="[\"${BARE_XP_PATH}/run0.fold0.nermodel\", \"${BARE_XP_PATH}/run0.fold1.nermodel\", \"${BARE_XP_PATH}/run0.fold2.nermodel\", \"${BARE_XP_PATH}/run0.fold3.nermodel\", \"${BARE_XP_PATH}/run0.fold4.nermodel\"]"
```


## Unsupervised Baselines

```sh
# we do not provide the full books for copyright reasons.
# To reproduce our experiments, you need to provide a
# directory with all the datasets books in txt format.
# The filenames must be the same as the filenames of the
# dekker dataset (see ./conivel/datas/dekker/dataset) with
# the .conll extension changed to .txt
EXTENDED_DOC_DIRS="directory_with_full_books_in_txt_format"

python xp_kfolds.py with runs/gen/book_bm25/config.json ner_model_paths="${NER_MODEL_PATHS}" cr_extended_doc_dirs="${EXTENDED_DOC_DIRS}"
python xp_kfolds.py with runs/gen/book_samenoun/config.json ner_model_paths="${NER_MODEL_PATHS}" cr_extended_doc_dirs="${EXTENDED_DOC_DIRS}"
python xp_kfolds.py with runs/gen/chapter_neighbors/config.json ner_model_paths="${NER_MODEL_PATHS}" cr_extended_doc_dirs="${EXTENDED_DOC_DIRS}"
```

To evaluate versions with retrieval on the first chapter only:

```sh
python xp_kfolds.py with runs/gen/chapter_bm25/config.json ner_model_paths="${NER_MODEL_PATHS}"
python xp_kfolds.py with runs/gen/chapter_samenoun/config.json ner_model_paths="${NER_MODEL_PATHS}"
```


## Generating a Context Retrieval Dataset

Alpaca 7b:

```sh
python xp_neural_context_retriever.py with runs/gen/genv3/config.json ner_model_paths="${NER_MODEL_PATHS}"
```

Alpaca 13b:

```sh
python xp_neural_context_retriever.py with runs/gen/genv3_13b/config.json ner_model_paths="${NER_MODEL_PATHS}"
```


## Neural Re-Ranker

```sh
# Depending on the context window, the value of n and of the model
# you are interested in
NEURAL_CONFIG="./runs/gen/neural_book_s7b_n24/config.json"

python xp_neural_context_retriever.py with "${NEURAL_CONFIG}" ner_model_paths="${NER_MODEL_PATHS}"
```


## Random Re-Rankers

`random re-ranker`:

```sh
python xp_random_reranker.py with './runs/gen/random_reranker_global/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```

`bucket random re-ranker`:

```sh
python xp_random_reranker.py with './runs/gen/random_reranker_bucket/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```


## MonoBERT and MonoT5

First, you need the pygaggle library:

>pip install pygaggle

You can then reproduce the results for the 4 configurations highlighted in the article.

`bm25+monobert`:

```sh
python xp_monoreranker.py with './runs/gen/monobert_bm25/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```

`all+monobert`:

```sh
python xp_monoreranker.py with './runs/gen/monobert_all/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```

`bm25+monot5`:

```sh
python xp_monoreranker.py with './runs/gen/monot5_bm25/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```

`all+monot5`:

```sh
python xp_monoreranker.py with './runs/gen/monot5_all/config.json' ner_model_paths="${NER_MODEL_PATHS}"
```


# Reproducing Plots

| Figure   | Script                      | Arguments       |
|----------|-----------------------------|-----------------|
| Figure 2 | `plot_n_comparison.py`      |                 |
| Figure 3 | `plot_mean_test_metrics.py` |                 |
| Figure 5 | `plot_mean_test_metrics.py` | `-g re-rankers` |
| Figure 4 | `plot_per_book_scores.py`   |                 |
| Figure 6 | `plot_chapter_vs_book.py`   |                 |
| Figure 7 | `plot_mean_test_metrics.py` | `-m precision`  |
| Figure 8 | `plot_mean_test_metrics.py` | `-m recall`     |

