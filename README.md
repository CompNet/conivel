# Conivel: CONtext In noVELs

## Installing dependencies

Dependencies are managed using [Poetry](https://python-poetry.org/). Use `poetry install` to install dependencies. You can then use `poetry shell` to obtain a shell with the created virtual environment activated.

Alternatively, you can manage you own virtual env and install dependencies manually with `pip` with the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```


# The Role of Global and Local Context in Named Entity Recognition

[arXiv link](https://arxiv.org/abs/2305.03132)

Please cite the article as follows :

```bibtex
@Article{amalvy-2023-context_ner,
      title={The Role of Global and Local Context in Named Entity Recognition}, 
      author={Amalvy, A. and Labatut, V. and Dufour, R.},
      year={2023},
      pages={2305.03132},
      journal={arXiv},
      volume={cs.CL},
}
```


## Dataset

The dataset (originally from [Dekker et al., 2019](https://github.com/Niels-Dekker/Out-with-the-Old-and-in-with-the-Novel/tree/master)) can be found under `conivel/datas/dekker/dataset`. It is using the simple CoNLL-2003 format. Our detailed annotation process can be found in the `annotation_process.pdf` file.


## Reproducing Results

For most of the experiments presented below, results can be found under the `runs/short` directory. If you reproduce an experiment as explained below, the Sacred library will create a new run under the `runs` directory. To plot results using the `plot_mean_test_metrics.py` script, runs must be placed into the `runs/short` directory with the correct name.

### No Retrieval

The `no retrieval` baseline for experiments found in Figure 1, 2 and 4 can be reproduced by using the following bash script:

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
	
# provided xp_bare.py creates a run with name "1"
mv runs/1 runs/short/bare
```


### Retrieval Heuristics


To reproduce the experiments presented in Figure 1, one can use:

```sh
#!/bin/bash

for heuristic in "left" "right" "neighbors" "random" "bm25" "samenoun"; do

    sents_nb_list="[1, 2, 3, 4, 5, 6]"
    # neighbors heuristc can only retrieve pair of sentence
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

    # provided you have no other runs in ./runs (the created run will
    # have name "1")
    # the following if statement correct for the historical names of
    # "left", "right" and "neighbors" heuristics
    if [[ "${heuristic}" = "left" ]]; then
	mv runs/1 runs/short/before
    elif [[ "${heuristic}" = "right" ]]; then
	mv runs/1 runs/short/after
    elif [[ "${heuristic}" = "neighbors" ]]; then
	mv runs/1 runs/short/surrounding
    else
	mv runs/1 "runs/short/${heuristic}"
    fi

done
```

The `plot_mean_test_metrics.py` script can then be used to plot Figure 1 of the paper.


### Oracle Versions of Retrieval Heuristics

Experiments found in Figure 2 can be reproduced with the following code:

```sh
#!/bin/bash

for heuristic in "left" "right" "neighbors" "random" "bm25" "samenoun"; do

    sents_nb_list="[1, 2, 3, 4, 5, 6]"
    # neighbors heuristc can only retrieve pair of sentence
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

    # provided you have no other runs in ./runs (the created run will
    # have name "1")
    # the following if statement correct for the historical names of
    # "left", "right" and "neighbors" heuristics
    if [[ "${heuristic}" = "left" ]]; then
	mv runs/1 runs/short/oracle_before
    elif [[ "${heuristic}" = "right" ]]; then
	mv runs/1 runs/short/oracle_after
    elif [[ "${heuristic}" = "neighbors" ]]; then
	mv runs/1 runs/short/oracle_surrounding
    else
	mv runs/1 "runs/short/oracle_${heuristic}"
    fi

done
```

`plot_mean_test_metrics.py -r` can be used to reproduce Figure 2. 


### Retrieved Sentences Distance Distribution

Experiments in Figure 3 can be reproduced used `xp_dist.py -r -o oracle_dists.json`, and the plot in the paper can then be reproduced with `plot_dist.py -i oracle_dists.json`.


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

# provided you have no runs in ./runs
mv runs/1 runs/short/bm25_restricted
```

The plot can be reproduced with `plot_mean_test_f1 -e`.


### Appendix: Dataset Details

Figure 5 can be reproduced using `plot_dekker_books_len.py`.
