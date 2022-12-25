"""Creating fragments takes a long time so we treat it as a pre-processing step."""
import logging

from gensim.models import Word2Vec
from cmn.fragments import create_noun_counts
from cmn.utils import conll2text

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    paths = ["data/input.conllu"]
    create_noun_counts(paths,
                       "output/nouns_restaurant.json")
    conll2text(paths, "output/all_txt_restaurant.txt")
    corpus = [x.lower().strip().split()
              for x in open("output/all_txt_restaurant.txt")]

    f = Word2Vec(corpus,
                 sg=0,
                 negative=5,
                 window=10,
                 vector_size=200,
                 min_count=2,
                 epochs=5,
                 workers=10)

    f.wv.save_word2vec_format(f"output/embeddings/restaurant_vecs_w2v.vec")
