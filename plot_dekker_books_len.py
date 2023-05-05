import argparse, os
import matplotlib.pyplot as plt
import scienceplots
from transformers import BertTokenizer  # type: ignore
from conivel.datas.dekker import DekkerDataset


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

dataset = DekkerDataset()

plt.style.use("science")
plt.xlabel("Number of sentences")
plt.ylabel("Number of books")
plt.hist([len(t) for t in dataset.documents])
if args.output:
    plt.savefig(os.path.expanduser(args.output))
else:
    plt.show()
