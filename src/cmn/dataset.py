"""Simple dataset loader for the 2014, 2015 semeval datasets."""
from sklearn.preprocessing import LabelEncoder
from functools import partial


def loader(instance_path,
           label_path,
           subset_labels,
           split_labels=False,
           mapping=None):
    subset_labels = set(subset_labels)
    labels = open(label_path)
    labels = [x.strip().lower().split() for x in labels]

    instances = []
    for line in open(instance_path):
        instances.append(line.strip().lower().split())

    if split_labels:
        labels = [[l.split("#")[0] for l in x] for x in labels]

    instances, gold = zip(*[(x, y[0]) for x, y in zip(instances, labels)
                            if len(y) == 1 and y[0]
                            in subset_labels])

    if mapping is not None:
        gold = [mapping.get(x, x) for x in gold]

    le = LabelEncoder()
    y = le.fit_transform(gold)
    label_set = le.classes_.tolist()

    return instances, y, label_set


citysearch_test = partial(loader,
                          instance_path="data/preprocessed_citysearch.txt",
                          label_path="data/preprocessed_citysearch_labels.txt",
                          subset_labels={"ambience",
                                         "service",
                                         "food"})


def restaurants_test():
    yield citysearch_test()
