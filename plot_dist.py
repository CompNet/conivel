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
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()


with open(args.input) as f:
    dists = json.load(f)

sn_dists = dists["samenoun_dists"]
bm25_dists = dists["bm25_dists"]


plt.style.use("science")
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(20, 4)
axs[0].hist(sn_dists, bins=50)
axs[0].set_xlabel("Distance of retrieved sentences (in sentences)", fontsize=20)
axs[0].set_title("samenoun", fontsize=20)
axs[1].hist(bm25_dists, bins=50)
axs[1].set_xlabel("Distance of retrieved sentences (in sentences)", fontsize=20)
axs[1].set_title("bm25", fontsize=20)
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
