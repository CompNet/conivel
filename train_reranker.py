import argparse, json
import torch
from sklearn.metrics import precision_recall_fscore_support
from conivel.datas.context import (
    BM25ContextRetriever,
    NeuralContextRetriever,
    ContextRetrievalDataset,
    ContextRetrievalExample,
)


def cr_dataset_from_path(path: str) -> ContextRetrievalDataset:
    with open(path) as f:
        data = json.load(f)
    return ContextRetrievalDataset([ContextRetrievalExample(**ex) for ex in data])


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--train-dataset",
    type=str,
    default="./runs/gen/genv3/fold0.cr_train_dataset.json",
)
parser.add_argument(
    "-e",
    "--test-dataset",
    type=str,
    default="./runs/gen/genv3/fold0.cr_test_dataset.json",
)
parser.add_argument("-b", "--batch-size", type=int, default=8)
parser.add_argument("-o", "--output", type=str)
args = parser.parse_args()

cr_train = cr_dataset_from_path(args.train_dataset)
cr_test = cr_dataset_from_path(args.test_dataset)


model = NeuralContextRetriever.train_context_selector(
    cr_train, 3, args.batch_size, 2e-5, dropout=0.1, huggingface_id="bert-base-cased"
)
model.save_pretrained(args.output)

# some of these parameters do not matter here
neural_retriever = NeuralContextRetriever(
    model, BM25ContextRetriever(sents_nb=8), args.batch_size, 1, threshold=0.5
)

raw_preds = neural_retriever.predict(cr_test)
preds = torch.argmax(raw_preds, dim=1).cpu()
labels = cr_test.labels()
assert not labels is None

precision, recall, f1, _ = precision_recall_fscore_support(
    labels, preds, average="micro"
)
print({"precision": precision, "recall": recall, "f1": f1})
