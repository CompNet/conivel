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

doc_tokens = []
for document in tqdm(dataset.documents):
    tokens = flattened([sent.tokens for sent in document])
    tokens = tokenizer.tokenize(" ".join(tokens))
    doc_tokens.append(tokens)


plt.style.use("science")
plt.xlabel("Number of tokens")
plt.hist([len(t) for t in doc_tokens])
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
