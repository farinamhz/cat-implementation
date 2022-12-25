"""Experiment on the test data."""
import json
import numpy as np

from cmn.simple import get_scores, attention, rbf_attention
from cmn.dataset import restaurants_test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter
from itertools import product


GAMMA = .03

if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("output/embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    d = json.load(open("output/nouns_restaurant.json"))

    nouns = Counter()
    for k, v in d.items():
        if k.lower() in r.items:
            nouns[k.lower()] += v

    embedding_paths = ["output/embeddings/restaurant_vecs_w2v.vec"]
    bundles = ((rbf_attention, attention), embedding_paths)
    aspects = [["ambience", "staff", "food"]]

    for att, path in product(*bundles):
        r = Reach.load(path, unk_word="<UNK>")

        for idx, (instances, y, label_set) in enumerate(restaurants_test()):

            s = get_scores(instances,
                           aspects,
                           r,
                           label_set,
                           gamma=GAMMA,
                           remove_oov=False,
                           attention_func=att)

            y_pred = s.argmax(1)
            f1_score = precision_recall_fscore_support(y, y_pred)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")
            scores[(att, path)][idx] = (f1_score, f1_macro)

    att_score = {k: v for k, v in scores.items() if k[0] == attention}
    att_per_class = [[z[x][0][:-1] for x in range(1)]
                     for z in att_score.values()]
    att_per_class = np.stack(att_per_class).mean(0)
    att_macro = np.mean([v[0][1][:-1] for v in att_score.values()], 0)

    rbf_score = {k: v for k, v in scores.items() if k[0] == rbf_attention}
    rbf_per_class = [[z[x][0][:-1] for x in range(1)]
                     for z in rbf_score.values()]
    rbf_per_class = np.stack(rbf_per_class).mean(0)
    rbf_macro = np.mean([v[0][1][:-1] for v in rbf_score.values()], 0)

    print("Attention per class :\t", att_per_class)
    print("RBF per class :\t", rbf_per_class)
