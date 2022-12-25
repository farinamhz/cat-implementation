"""Grid search over the test data."""
import numpy as np
import pandas as pd
import json

# import sys
# sys.path.insert(0, 'cat-implementation/cat/simple')

# import sys
# print("sys", sys.path)
# sys.path.append("cat-implementation/")

from src.cmn.simple import (get_scores,
                            attention,
                            rbf_attention,
                            mean)
from src.cmn.dataset import restaurants_test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
from itertools import product
from tqdm import tqdm

if __name__ == "__main__":

    scores = defaultdict(dict)

    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    nouns = json.load(open("data/nouns_restaurant.json"))

    aspects = [["ambience", "staff", "food"]]

    gamma = np.arange(.0, .1, .01)
    attentions = [(-1, attention), (-1, mean)]
    attentions.extend(product(gamma, [rbf_attention]))

    fun2name = {attention: "att", mean: "mean", rbf_attention: "rbf"}
    pbar = tqdm(total=(len(attentions)))

    df = []
    datas = list(restaurants_test())

    for g, att_func in attentions:
        if att_func == rbf_attention:
            r.vectors[r.items["<UNK>"]] += 10e5
        else:
            r.vectors[r.items["<UNK>"]] *= 0

        for idx, (inst,
                  y,
                  label_set) in enumerate(datas):
            s = get_scores(inst,
                           aspects,
                           r,
                           label_set,
                           gamma=g,
                           remove_oov=False,
                           attention_func=att_func)

            y_pred = s.argmax(1)
            # f1_score = precision_recall_fscore_support(y, y_pred)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")[:-1]
            row = (g,
                   fun2name[att_func],
                   3,
                   *f1_macro)
            df.append(row)
        pbar.update(1)
    df = pd.DataFrame(df, columns=("gamma",
                                   "function",
                                   "n_noun",
                                   "p",
                                   "r",
                                   "f1"))
    df.to_csv("results_grid_search.csv")
