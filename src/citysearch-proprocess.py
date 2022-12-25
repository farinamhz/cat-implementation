
with open("../data/test_label.txt", "r") as f:
    candidate_labels = []
    candidate_sentences_ids = []
    for i, line in enumerate(f.readlines()):
        aspects = line.split()
        if len(aspects) == 1 and aspects[0] in ["Food", "Ambience", "Staff"]:
            candidate_sentences_ids.append(i)
            candidate_labels.append(line)

with open("../data/test.txt", "r") as f:
    sentences = f.readlines()

with open("../data/preprocessed_citysearch.txt", "w") as f:
    for i in candidate_sentences_ids:
        f.write(sentences[i].lstrip())

with open("../data/preprocessed_citysearch_labels.txt", "w") as f:
    for label in candidate_labels:
        f.write(label)