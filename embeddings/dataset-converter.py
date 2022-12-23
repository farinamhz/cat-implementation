import stanza
from stanza.utils.conll import CoNLL
import spacy
import argparse

conll = [[['1', 'Test', '_', 'NOUN', 'NN', 'Number=Sing','0', '_', '_', 'start_char=0|end_char=4'],
          ['2', 'sentence', '_', 'NOUN', 'NN', 'Number=Sing', '1', '_', '_', 'start_char=5|end_char=13'],
          ['3', '.', '_', 'PUNCT', '.', '_', '2', '_', '_', 'start_char=13|end_char=14']]]

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='data', type=str, default='/cat-implementation/data/train.txt', help='dataset file path')
parser.add_argument('--output', dest='output', type=str, default='/cat-implementation/data/train.txt', help='output path')
args = parser.parse_args()

# nlp = spacy.load("en_core_web_sm")


stanza.download('en') # download English model
nlp = stanza.Pipeline('en') # initialize English neural pipeline
doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence

try:

    with open(args.data, encoding='utf8') as inp:
        for doc in nlp.pipe(inp, batch_size=1000, disable=['ner']):
        CoNLL.write_doc2conll(doc, args.output)
except (FileNotFoundError, EOFError) as e:
    print(e)
# finally:
#     if args.output is not None:
#         args.output.close()



