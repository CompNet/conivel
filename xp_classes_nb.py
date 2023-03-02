import argparse
from conivel.datas.conll import CoNLLDataset
from conivel.datas.datas import NERSentence
from conivel.datas.dekker import DekkerDataset
from conivel.datas.ontonotes import OntonotesDataset
from conivel.predict import predict
from conivel.score import score_ner
from conivel.utils import pretrained_bert_for_token_classification
from conivel.train import train_ner_model


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train-dataset", type=str)
parser.add_argument("-p", "--dataset-path", type=str)
args = parser.parse_args()


if args.train_dataset == "conll":
    train = CoNLLDataset.train_dataset()
    train_only_per = CoNLLDataset.train_dataset(keep_only_classes={"PER"})
elif args.train_dataset == "ontonotes":
    train = OntonotesDataset(args.dataset_path)
    train_only_per = OntonotesDataset(args.dataset_path)
    for doc in train_only_per.documents:
        for sent in doc:
            sent = NERSentence(
                sent.tokens,
                [t if t in ["B-PER", "I-PER", "O"] else "O" for t in sent.tags],
            )
else:
    raise RuntimeError


test = DekkerDataset(keep_only_classes={"PER"})

scores = {}

for train_name, local_train in [("all classes", train), ("only per", train_only_per)]:
    ner_model = pretrained_bert_for_token_classification(
        "bert-base-cased", local_train.tag_to_id
    )
    ner_model = train_ner_model(ner_model, local_train, local_train, epochs_nb=2)
    preds = predict(ner_model, test).tags
    scores[train_name] = score_ner(test.sents(), preds)

print(scores)
