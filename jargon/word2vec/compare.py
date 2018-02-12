#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import argparse
import os
import logging
import gensim
import time

from numpy import load
from monster.misc import get_res_filepath
from ..utils.math import cosine, bhattacharyya

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def compare(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for prediction")
    parser.add_argument("-p", "--prob", help="prob file", type=str)
    parser.add_argument("-c", "--occur", help="occur file", type=str)
    parser.add_argument(
        "-m", "--model", help="either good or bad model", type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename",
        type=str,
        default="compared.json")

    options = options = parser.parse_args(args)

    if options.prob and os.path.isfile(options.prob):
        probability = load(options.prob)
        logging.info("Prob file '{}' loaded".format(options.prob))
    else:
        logging.error("Error: prob file '{}' not found".format(options.prob))
        return 1

    if options.occur and os.path.isfile(options.occur):
        occurrence = load(options.occur)
        logging.info("Occur file '{}' loaded".format(options.occur))
    else:
        logging.error("Error: occur file '{}' not found".format(options.occur))
        return 1

    if options.model and os.path.isfile(options.model):
        model = gensim.models.Word2Vec.load(options.model)
    else:
        logging.error("Error: model file '{}' not found".format(options.model))
        return 1

    compare_impl(probability, occurrence, model, options.output)


def compare_impl(probability, occurrence, model, output):
    res = dict()
    logging.info("Start comparing...")
    start_ts = time.time()
    for row_idx, row in enumerate(probability):
        word = model.wv.index2word[row_idx]
        top_prob_indices = gensim.matutils.argsort(row, topn=40, reverse=True)
        top_occur_indices = gensim.matutils.argsort(
            occurrence[row_idx], topn=40, reverse=True)
        top_prediction = [(model.wv.index2word[index1], float(row[index1]))
                          for index1 in top_prob_indices]
        top_occurrence = [(model.wv.index2word[index1],
                           float(occurrence[row_idx][index1]))
                          for index1 in top_occur_indices]
        res[word] = dict()
        res[word]['most_probable'] = top_prediction
        res[word]['most_occurred'] = top_occurrence
        # res[word]['bhattacharyya'] = bhattacharyya(row, occurrence[row_idx])
        res[word]['cosine'] = cosine(row, occurrence[row_idx])
        if row_idx == len(probability) - 1 or row_idx % 100 == 0:
            current_ts = time.time()
            logging.info(
                "Processed_words: {:d} Progress: {:.02%}  Words/sec: {:.02f}".
                format(row_idx, row_idx / len(probability), row_idx / (
                    current_ts - start_ts)))

    outfile = get_res_filepath(output)
    json.dump(res, open(outfile, 'w'), indent=2)
    logging.info("Job finished, results saved at '{}'".format(outfile))
