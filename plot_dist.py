import argparse, json, os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-o", "--output", type=str, default=None)
args = parser.parse_args()


with open(args.input) as f:
    dists = json.load(f)

sn_dists = dists["samenoun_dists"]
bm25_dists = dists["bm25_dists"]


plt.style.use("science")
plt.rc("xtick", labelsize=30)  # fontsize of the tick labels
plt.rc("ytick", labelsize=30)  # fontsize of the tick labels
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
fig.set_size_inches(20, 6)
axs[0].hist(sn_dists, bins=50)
axs[0].set_title("samenoun", fontsize=30)
axs[1].hist(bm25_dists, bins=50)
axs[0].set_ylabel("Number of retrieved sentences", fontsize=25)
fig.supxlabel(
    "Distance between the input sentence and the retrieved sentence, in number of sentences",
    fontsize=30,
)
axs[1].set_title("bm25", fontsize=30)
if args.output:
    plt.savefig(os.path.expanduser(args.output))
else:
    plt.show()
