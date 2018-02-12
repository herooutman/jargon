#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import gensim

from threading import Thread
from numpy import exp, dot, save, zeros, sum as np_sum

from monster.misc import get_res_filepath

from .compare import compare_impl

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def predict(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for prediction")
    parser.add_argument("-g", "--good", help="good model", type=str)
    parser.add_argument("-b", "--bad", help="bad model", type=str)
    parser.add_argument(
        "-o", "--output", help="output filename", type=str, default="word2vec")

    options = options = parser.parse_args(args)

    if options.good and os.path.isfile(options.good):
        good_model = gensim.models.Word2Vec.load(options.good)
    else:
        logging.error("Error: good model file '{}' not found".format(options.good))
        return 1

    if options.bad and os.path.isfile(options.bad):
        bad_model = gensim.models.Word2Vec.load(options.bad)
    else:
        logging.error("Error: bad model file '{}' not found".format(options.bad))
        return 1
    predict_impl(good_model, bad_model, options.output)


def predict_impl(good_model, bad_model, output):
    logging.info("start calculating probability")
    if not good_model.negative or not bad_model.negative:
        raise RuntimeError(
            "We have currently only implemented predict_output_word for the negative sampling scheme, "
            "so you need to have run word2vec with negative > 0 for this to work."
        )

    if not hasattr(bad_model.wv, 'syn0') or not hasattr(good_model, 'syn1neg'):
        raise RuntimeError(
            "Parameters required for predicting the output words not found.")

    syn0 = bad_model.wv.syn0
    syn1 = good_model.syn1neg

    probability = exp(dot(syn0, syn1.T))
    rows, columns = probability.shape
    logging.info(
        "probability matrix rows: {}, columns: {}".format(rows, columns))
    sums = np_sum(probability, axis=1)
    logging.info("probability sum matrix shape: {}".format(sums.shape))

    probability = probability / sums[:, None]
    logging.info("probability calculation finished")
    pred_outfile = get_res_filepath(fn="{}.prob.npy".format(output))
    t1 = Thread(target=save, args=(pred_outfile, probability))
    t1.start()

    # logging.info("start occurrence counting")
    # occurrence = zeros((rows, rows))
    # # TODO
    # logging.info("occurrence counting finished")
    # occur_outfile = get_res_filepath(fn="{}.occur.npy".format(output))
    # t2 = Thread(target=save, args=(occur_outfile, occurrence))
    # save(occur_outfile, occurrence)
    # compare_outfile = get_res_filepath(fn="{}.compare.json".format(output))
    # compare_impl(probability, occurrence, bad_model, compare_outfile)

    t1.join()
    logging.info("prediction results saved at '{}'".format(pred_outfile))
    # t2.join()
    # logging.info("occurrence results saved at '{}'".format(occur_outfile))
