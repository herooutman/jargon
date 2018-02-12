#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
import logging
import gensim

from collections import Counter

from gensim.models.word2vec import LineSentence
from monster.misc import get_res_filepath

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def train(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for training")
    parser.add_argument("-g", "--good", help="good corpus file", type=str)
    parser.add_argument("-b", "--bad", help="bad corpus file", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename",
        type=str,
        default="word2vec.model")
    parser.add_argument(
        "-s", "--size", help="word vector size", type=int, default=100)
    parser.add_argument(
        "-a",
        "--alpha",
        help="initial learning rate",
        type=float,
        default=0.025)
    parser.add_argument(
        "-w", "--window", help="window size", type=int, default=5)
    parser.add_argument(
        "-S", "--sample", help="subsampling rate", type=float, default=1e-3)
    parser.add_argument(
        "-T", "--threads", help="thread number", type=int, default=3)
    parser.add_argument(
        "--min_alpha",
        help="minimal learning rate",
        type=float,
        default=0.0001)
    parser.add_argument(
        "--sg", help="skip gram (1) or cbow (0)", type=int, default=0)
    parser.add_argument(
        "--hs",
        help="using hierarchical softmax (1) or not (0)",
        type=int,
        default=0)
    parser.add_argument(
        "-n", "--negative", help="negative sampling", type=int, default=5)
    parser.add_argument("--cbow_mean", help="cbow mean", type=int, default=1)
    parser.add_argument("-i", "--iter", help="iterations", type=int, default=5)
    parser.add_argument(
        "--min_count",
        help="minimal occurrence of words to be considered",
        type=int,
        default=5)

    options = options = parser.parse_args(args)
    vocab = dict()

    logging.info("loading corpus...")
    if options.good and os.path.isfile(options.good):
        good_sentences = list(
            LineSentence(options.good, max_sentence_length=10000))
        vocab['good'] = get_vocab(options.good)
    else:
        logging.error("Error: good corpus file '{}' not found".format(options.good))
        return 1

    if options.bad and os.path.isfile(options.bad):
        bad_sentences = list(
            LineSentence(options.bad, max_sentence_length=10000))
        vocab['bad'] = get_vocab(options.bad)

    else:
        bad_sentences = list()

    min_count = options.min_count

    good_outfile = get_res_filepath(fn="{}.good.model".format(options.output))
    bad_outfile = get_res_filepath(fn="{}.bad.model".format(options.output))
    vocab_outfile = get_res_filepath(fn="{}.vocab".format(options.output))

    with open(vocab_outfile, "w") as fd:
        json.dump(vocab, fd)

    good_model = gensim.models.Word2Vec(
        workers=options.threads,
        window=options.window,
        negative=options.negative,
        sg=options.sg,
        size=options.size,
        sample=options.sample,
        min_count=min_count,
        iter=options.iter,
        alpha=options.alpha,
        min_alpha=options.min_alpha,
        hs=options.hs,
        cbow_mean=options.cbow_mean, )

    good_model.build_vocab(good_sentences + bad_sentences)
    good_model.train(
        good_sentences,
        total_examples=len(good_sentences),
        epochs=good_model.iter)
    good_model.save(good_outfile)

    if bad_sentences:
        bad_model = gensim.models.Word2Vec(
            workers=options.threads,
            window=options.window,
            negative=options.negative,
            sg=options.sg,
            size=options.size,
            sample=options.sample,
            min_count=min_count,
            iter=options.iter,
            alpha=options.alpha,
            min_alpha=options.min_alpha,
            hs=options.hs,
            cbow_mean=options.cbow_mean, )

        bad_model.build_vocab(good_sentences + bad_sentences)
        bad_model.train(
            bad_sentences,
            total_examples=len(bad_sentences),
            epochs=bad_model.iter)
        bad_model.save(bad_outfile)


def get_vocab(train_file):
    with open(train_file) as fd:
        texts = fd.read()
        words = texts.split()
        c = Counter(words)
        vocab = dict(c)
    return vocab
