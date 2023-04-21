import argparse
import matplotlib.pyplot as plt
import scienceplots
from transformers import BertTokenizer  # type: ignore
from tqdm import tqdm
from conivel.datas.dekker import DekkerDataset
from conivel.utils import flattened


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

dataset = DekkerDataset()

plt.style.use("science")
plt.xlabel("Number of sentences")
plt.hist([len(t) for t in dataset.documents])
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
