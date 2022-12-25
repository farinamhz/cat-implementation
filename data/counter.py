unique = set()
with open("../data/train.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        for a in line.split():
            unique.add(a)
print(len(unique))