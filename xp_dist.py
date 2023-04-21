import argparse, json
import matplotlib.pyplot as plt
from tqdm import tqdm
from conivel.datas.context import (
    SameNounRetriever,
    BM25ContextRetriever,
    IdealNeuralContextRetriever,
)
from conivel.datas.dekker import DekkerDataset
from conivel.utils import pretrained_bert_for_token_classification
from conivel.train import train_ner_model


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str)
parser.add_argument("-r", "--oracle", action="store_true")
args = parser.parse_args()


sn_dists = []
bm25_dists = []

dataset = DekkerDataset()
kfolds = dataset.kfolds(5, shuffle=True, shuffle_seed=0)

for train, test in kfolds:

    # * retriever instantiation
    if args.oracle:
        ner_model = pretrained_bert_for_token_classification(
            "bert-base-cased", dataset.tag_to_id
        )
        ner_model = train_ner_model(
            ner_model, train, train, epochs_nb=2, learning_rate=2e-5
        )
        sn_retriever = IdealNeuralContextRetriever(
            1, SameNounRetriever(16), ner_model, 4, dataset.tags
        )
        bm25_retriever = IdealNeuralContextRetriever(
            1, BM25ContextRetriever(16), ner_model, 4, dataset.tags
        )
    else:
        sn_retriever = SameNounRetriever(1)
        bm25_retriever = BM25ContextRetriever(1)

    # * retrieval
    for document in tqdm(test.documents):  # TODO
        for sent_i, sent in enumerate(document):
            sn_matchs = sn_retriever.retrieve(sent_i, document)
            bm25_matchs = bm25_retriever.retrieve(sent_i, document)
            if len(sn_matchs) != 0:
                sn_dists.append(abs(sent_i - sn_matchs[0].sentence_idx))
            bm25_dists.append(abs(sent_i - bm25_matchs[0].sentence_idx))


with open(args.output, "w") as f:
    json.dump({"samenoun_dists": sn_dists, "bm25_dists": bm25_dists}, f, indent=4)
