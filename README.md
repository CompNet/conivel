# Conivel: CONtext In noVELs

## Installing dependencies

Use `poetry install` to install dependencies. You can then use `poetry shell` to obtain a shell with the created virtual environment activated.


# The Role of Global and Local Context in Named Entity Recognition

## Reproducing Results

For all the scripts presented below, results can be found under the `runs` directory. To be able to reproduce plots for Figure 1, 2 and 4, results must be placed under `runs/short/` and named correctly.

### No Retrieval

The `no retrieval` baseline for experiments found in Figure 1, 2 and 4
can be reproduced by using the following bash script:

```sh
#!/bin/bash 

python xp_bare.py with\
	k=5\
	shuffle_kfolds_seed=0\
	batch_size=8\
	save_models=False\
	runs_nb=3\
	ner_epochs_nb=2\
	ner_lr=2e-5\
	dataset_name="dekker"
```


### Retrieval Heuristics


To reproduce the experiments presented in Figure 1, one can use:

```sh
#!/bin/bash

for heuristic in "left" "right" "neighbors" "random" "bm25" "samenoun"; do

    sents_nb_list="[1, 2, 3, 4, 5, 6]"
    if [[ "${heuristic}" = "neighbors" ]]; then
	sents_nb_list="[2, 4, 6]"
    fi

    python xp_kfolds.py with\
		k=5\
		shuffle_kfolds_seed=0\
		batch_size=8\
		save_models=False\
		runs_nb=3\
		context_retriever="${heuristic}"\
		context_retriever_kwargs='{}'\
		sents_nb_list="${sents_nb_list}"\
		ner_epochs_nb=2\
		ner_lr=2e-5\
		dataset_name="dekker"

done
```

The `plot_mean_test_f1.py` can then be used to plot the curves found
in the paper.


### Oracle Versions of Retrieval Heuristics

Experiments found in Figure 2 can be reproduced with the following code:

```sh
#!/bin/bash

for heuristic in "left" "right" "neighbors" "random" "bm25" "samenoun"; do

    sents_nb_list="[1, 2, 3, 4, 5, 6]"
    if [[ "${heuristic}" = "neighbors" ]]; then
	sents_nb_list="[2, 4, 6]"
    fi

    python xp_ideal_neural_retriever.py with\
	    k=5\
	    shuffle_kfolds_seed=0\
	    batch_size=8\
	    save_models=False\
	    runs_nb=3\
	    retrieval_heuristic="${heuristic}"\
	    retrieval_heuristic_inference_kwargs='{"sents_nb": 16}'\
	    sents_nb_list="${sents_nb_list}"\
	    ner_epochs_nb=2\
	    ner_lr=2e-5

done
```

`plot_mean_test_f1.py -r` can be used to reproduce Figure 2. 


### Retrieved Sentences Distance Distribution

Experiments in Figure 3 can be reproduced used `xp_dist.py -r -o dists.json`, and the plot in the paper can then be reproduced with `plot_dist.py -i dists.json`.


### Restricted BM25 heuristic

To reproduce the experiment found in Figure 4, use:

```sh
python xp_ideal_neural_retriever.py with\
	k=5\
	shuffle_kfolds_seed=0\
	batch_size=8\
	save_models=False\
	runs_nb=3\
	retrieval_heuristic="bm25_restricted"\
	retrieval_heuristic_inference_kwargs='{"sents_nb": 16}'\
	sents_nb_list='[1,2,3,4,5,6]'\
	ner_epochs_nb=2\
	ner_lr=2e-5
```

The plot can be reproduced with `plot_mean_test_f1 -e`.


### Appendix: Dataset Details

Figure 5 can be reproduced using `plot_dekker_books_len.py`.
