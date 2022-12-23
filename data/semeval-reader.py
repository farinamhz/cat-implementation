import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ParseError
from typing import List, Union, Dict, Any, Optional


# import tempfile
# import zipfile
# import tarfile
#
# from allennlp.common.file_utils import cached_path
# import requests

# from target_extraction.data_types import TargetTextCollection, TargetText
# from target_extraction.data_types_util import Span


def semeval_2014(data_fp, conflict):
    '''
    The sentiment labels are the following: 1. negative, 2. neutral,
    3. positive, and 4. conflict. conflict will not appear if the argument
    `conflict` is False.
    :param data_fp: Path to the SemEval 2014 formatted file.
    :param conflict: Whether or not to include targets or categories that
                     have the `conflict` sentiment value. True is to include
                     conflict targets and categories.
    :returns: The SemEval 2014 data formatted into a
              `target_extraction.data_types.TargetTextCollection` object.
    :raises SyntaxError: If the File passed is detected as not a SemEval
                         formatted file.
    :raises `xml.etree.ElementTree.ParseError`: If the File passed is
                                                not formatted correctly e.g.
                                                mismatched tags
    '''

    tree = ET.parse(data_fp)
    sentences = tree.getroot()
    if sentences.tag != 'sentences':
        raise SyntaxError('The root of all semeval xml files should '
                          f'be sentences and not {sentences.tag}')
    return _semeval_extract_data(sentences, conflict)


def _semeval_extract_data(sentence_tree, conflict):
    '''
    :param sentence_tree: The root element of the XML tree that has come
                          from a SemEval XML formatted XML File.
    :param conflict: Whether or not to include targets or categories that
                     have the `conflict` sentiment value. True is to include
                     conflict targets and categories.
    :returns: The SemEval data formatted into a
              `target_extraction.data_types.TargetTextCollection` object.
    '''
    all_text = []
    for sentence in sentence_tree:
        text_id = sentence.attrib['id']
        targets = []
        target_sentiments = []

        category_sentiments = []
        categories = []

        for data in sentence:
            if data.tag == 'text':
                text = data.text
                text = text.replace(u'\xa0', u' ')
                all_text.append(text)

            elif data.tag == 'aspectTerms':
                for target in data:
                    # If it is a conflict sentiment and conflict argument True skip this target
                    target_sentiment = target.attrib['polarity']
                    if not conflict and target_sentiment == 'conflict':
                        continue
                    targets.append(target.attrib['term'].replace(u'\xa0', u' '))
                    target_sentiments.append(target_sentiment)
                    span_from = int(target.attrib['from'])
                    span_to = int(target.attrib['to'])

            elif data.tag == 'aspectCategories':
                for category in data:
                    # If it is a conflict sentiment and conflict argument True skip this category
                    category_sentiment = category.attrib['polarity']
                    if not conflict and category_sentiment == 'conflict':
                        continue
                    categories.append(category.attrib['category'])
                    category_sentiments.append(category.attrib['polarity'])
                if len(categories) != 1:
                    print(all_text.pop(), categories)
                    # all_text.pop()
                    break
                # if "food" not in categories and "service" not in categories and \
                #         "ambience" not in categories and "price" not in categories and \
                #         "anecdotes/miscellaneous" not in categories:
                #     all_text.pop()
                #     break
            elif data.tag == 'Opinions':
                for opinion in data:
                    category_target_sentiment = opinion.attrib['polarity']
                    if not conflict and category_target_sentiment == 'conflict':
                        continue
                    # Handle the case where some SemEval 16 files do
                    # not contain targets and are only category sentiment files
                    if 'target' in opinion.attrib:
                        # Handle the case where there is a category but no target
                        target_text = opinion.attrib['target'].replace(u'\xa0', u' ')
                        span_from = int(opinion.attrib['from'])
                        span_to = int(opinion.attrib['to'])
                        # Special cases for poor annotation in SemEval 2016
                        # task 5 subtask 1 Restaurant dataset
                        if text_id == 'DBG#2:15' and target_text == 'NULL':
                            span_from = 0
                            span_to = 0
                        if text_id == "en_Patsy'sPizzeria_478231878:2" \
                                and target_text == 'NULL':
                            span_to = 0
                        if text_id == "en_MercedesRestaurant_478010602:1" \
                                and target_text == 'NULL':
                            span_to = 0
                        if text_id == "en_MiopostoCaffe_479702043:9" \
                                and target_text == 'NULL':
                            span_to = 0
                        if text_id == "en_MercedesRestaurant_478010600:1" \
                                and target_text == 'NULL':
                            span_from = 0
                            span_to = 0
                        if target_text == 'NULL':
                            target_text = None
                            # Special cases for poor annotation in SemEval 2016
                            # task 5 subtask 1 Restaurant dataset
                            if text_id == '1490757:0':
                                target_text = 'restaurant'
                            if text_id == 'TR#1:0' and span_from == 27:
                                target_text = 'spot'
                            if text_id == 'TFS#5:26':
                                target_text = "environment"
                            if text_id == 'en_SchoonerOrLater_477965850:10':
                                target_text = 'Schooner or Later'
                        targets.append(target_text)
                    categories.append(opinion.attrib['category'])
                    target_sentiments.append(category_target_sentiment)
    return all_text


if __name__ == "__main__":
    result = semeval_2014("Restaurants_Test_Data_PhaseA.xml", True)
    print(len(result))
