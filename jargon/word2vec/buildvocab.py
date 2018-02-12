#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import gensim
# import numpy as np

from gensim.models.word2vec import LineSentence
from monster.misc import get_res_filepath

# DATA_DIR = os.path.abspath(
#     os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def build_vocab_impl(input, output, min_count):
    # print(options)
    # print(type(options))
    if input and os.path.isfile(input):
        sentences = LineSentence(input, max_sentence_length=10000)
    else:
        print("Error: input file '{}' not found".format(input))
        return 1

    outfile = get_res_filepath(fn=output)
    #  -cbow 0 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
    model = gensim.models.Word2Vec(
        min_count=min_count)
    model.build_vocab(sentences=sentences)

    model.save(outfile)


def build_vocab(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for training")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename",
        type=str,
        default="word2vec.model")
    parser.add_argument(
        "-m",
        "--min_count",
        help="minimal occurrence of words to be considered",
        type=int,
        default=5)

    options = options = parser.parse_args(args)
    build_vocab_impl(**vars(options))
