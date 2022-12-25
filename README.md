# cat

This is the repository for the ACL 2020 paper [Embarrassingly Simple Unsupervised Aspect Extraction](https://www.aclweb.org/anthology/2020.acl-main.290/).
In this work, we extract aspects from restaurant reviews with attention that uses RBF kernels.

## Setup
It has been developed on `Python 3.8` and can be installed by `pip`:

```bash
git clone https://github.com/farinamhz/cat-implementation.git
cd cat-implementation
pip install -r requirements.txt
```
Additionally, you need to install the following library from the source:
- ``en_core_web_sm`` as a requirement in ``spaCy`` library with the following command:
  
  ```bash
  python -m spacy download en_core_web_sm
  ```

## Quickstart

### Data

In the paper,SemEval 2014, 2015 and citysearch dataset are used, which you can download from the following links:

* [semeval 2014](http://alt.qcri.org/semeval2014/task4/)
* [semeval 2015](http://alt.qcri.org/semeval2015/task12/)
* [citysearch](https://www.cs.cmu.edu/~mehrbod/RR/) (we used the link from [this repository](https://github.com/ruidan/Unsupervised-Aspect-Extraction))
 
 However, they are available in the[`./data`](./data) directory.


### Preprocess

1- Preprocess the CitySearch dataset to exclude any review with more than one aspect or aspect other than Food, Staff, or Ambience.
```bash
python src\citysearch-proprocess.py
```
2- Converting the CitySearch dataset, which is in text format, to CoNLLu format:
```bash
python src\text-to-CoNLLu.py data\train.txt data\input.conllu
```
3- Training in-domain word embeddings on train subset of CitySearch dataset:
```bash
python src\preprocessing_embeddings.py
```

### Run and Results
Experiment on test dataset (CitySearch) which is already preprocessed in the previous steps and see results of gridsearch with different values for gamma parameter and attention, mean, and rbf as the attention functions with three aspects which are Food, Staff, and Ambience.

- Experiment per class for all three classes:
    ```bash
    python src\experiment_perclass.py
    ```
  The results will be shown in console.

- Experiment with macro scores in total:
    ```bash
    python src\experiment.py
    ```
  The result will be saved in [`./output`](./output) directory with this name: results_grid_search.csv

## Citing

```bibtex
@inproceedings{tulkens2020embarrassingly,
    title = "Embarrassingly Simple Unsupervised Aspect Extraction",
    author = "Tulkens, St{\'e}phan  and  van Cranenburgh, Andreas",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.290",
    doi = "10.18653/v1/2020.acl-main.290",
    pages = "3182--3187",
}
```