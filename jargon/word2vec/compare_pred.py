#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import argparse
import os
import logging
import gensim
import time
import math
import numpy as np

from threading import Thread
from scipy.stats import spearmanr
from monster.misc import get_res_filepath
from monster.atomic import AtomicCounter

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "compare 2 predictions"))


def compare_pred(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for prediction")
    parser.add_argument("--prob1", help="prob1 file", type=str)
    parser.add_argument("--prob2", help="prob2 file", type=str)
    parser.add_argument("--prob3", help="prob3 file", type=str)
    parser.add_argument("-m", "--model", help="eithor good or bad model", type=str)
    parser.add_argument(
        "-t", "--thread", help="number of threads", type=int, default=10)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename",
        type=str,
        default="compare_pred.json")

    options = options = parser.parse_args(args)
    if options.prob1 and os.path.isfile(options.prob1):
        logging.info("Loading prob1 file")
        probability1 = np.load(options.prob1)
        logging.info("Prob1 file '{}' loaded".format(options.prob1))
    else:
        logging.error("Error: prob1 file '{}' not found".format(options.prob1))
        return 1

    if options.prob2 and os.path.isfile(options.prob2):
        logging.info("Loading prob2 file")
        probability2 = np.load(options.prob2)
        logging.info("Prob2 file '{}' loaded".format(options.prob2))
    else:
        logging.error("Error: prob2 file '{}' not found".format(options.prob2))
        return 1

    if options.prob3 and os.path.isfile(options.prob3):
        logging.info("Loading prob3 file")
        probability3 = np.load(options.prob3)
        logging.info("Prob3 file '{}' loaded".format(options.prob3))
    else:
        logging.error("Error: prob3 file '{}' not found".format(options.prob3))
        return 1

    if options.model and os.path.isfile(options.model):
        model = gensim.models.Word2Vec.load(options.model)
    else:
        logging.error("Error: model file '{}' not found".format(options.model))
        return 1

    compare_pred_impl(probability1, probability2, probability3, model,
                      options.output, options.thread)


def jaccard_index(l1, l2, topn=None):
    if topn:
        l1 = l1[:topn]
        l2 = l2[:topn]
    t = len(set(l1).intersection(set(l2)))
    return t / (len(l1) + len(l2) - t)


def compare_pred_thread(res, p1, p2, p3, model, start, end, start_ts,
                        progress):
    logging.info("Starting thread, data range: {} - {}.".format(start, end))
    for idx in range(start, min(end, len(p1))):
        # v1 = gensim.matutils.argsort(p1[idx], topn=len(p1[idx]), reverse=True)
        # v1: gb
        # v2: gg
        # v3: bb
        word = model.wv.index2word[idx]
        v1 = gensim.matutils.argsort(p1[idx], topn=1000, reverse=True)
        d1 = [float(p1[idx][x]) for x in v1]
        v2 = gensim.matutils.argsort(p2[idx], topn=1000, reverse=True)
        d2 = [float(p1[idx][x]) for x in v2]
        v3 = gensim.matutils.argsort(p3[idx], topn=1000, reverse=True)
        d3 = [float(p1[idx][x]) for x in v3]

        corr, pv = spearmanr(v1, v2)
        res[word] = dict()
        res[word]['occurrence'] = model.wv.vocab[word].count
        res[word]['correlation'] = corr
        res[word]['pvalue'] = pv
        res[word]['jac40'] = jaccard_index(v1, v2, 40), jaccard_index(
            v3, v2, 40)
        res[word]['jac100'] = jaccard_index(v1, v2, 100), jaccard_index(
            v3, v2, 100)
        res[word]['jac1000'] = jaccard_index(v1, v2, 1000), jaccard_index(
            v3, v2, 1000)

        res[word]['prob_std1'] = np.std(d1)
        res[word]['prob_std2'] = np.std(d2)
        res[word]['prob_std3'] = np.std(d3)

        progress.increment()
        if progress.value() % 1000 == 0:
            current_ts = time.time()
            logging.info(
                "Processed_words: {:d} Progress: {:.02%}  Words/sec: {:.02f}".
                format(progress.value(),
                       progress.value() / len(p1),
                       progress.value() / (current_ts - start_ts)))
    logging.info("Thread exit.")


def stat_dict(l):
    return dict(
        mean=np.mean(l),
        max=np.max(l),
        min=np.min(l),
        median=np.median(l),
        std=np.std(l))


def compare_pred_impl(p1, p2, p3, model, output, threads_n):
    progress = AtomicCounter()
    res = dict()
    logging.info("Start comparing...")
    start_ts = time.time()
    threads = []
    batch = math.ceil(len(p1) / threads_n)
    for i in range(threads_n):
        t = Thread(
            target=compare_pred_thread,
            args=(res, p1, p2, p3, model, batch * i, batch + batch * i,
                  start_ts, progress))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    corrs = [x['correlation'] for x in res.values()]
    pvs = [x['pvalue'] for x in res.values()]
    jac40 = list(zip(* [x['jac40'] for x in res.values()]))
    jac100 = list(zip(* [x['jac100'] for x in res.values()]))
    jac1000 = list(zip(* [x['jac1000'] for x in res.values()]))
    prob_std1 = [x['prob_std1'] for x in res.values()]
    prob_std2 = [x['prob_std2'] for x in res.values()]
    prob_std3 = [x['prob_std3'] for x in res.values()]

    stats = dict()
    stats["correlation"] = stat_dict(corrs)
    stats["pvalue"] = stat_dict(pvs)
    stats["jac40"] = stat_dict(jac40[0]), stat_dict(jac40[1])
    stats["jac100"] = stat_dict(jac100[0]), stat_dict(jac100[1])
    stats["jac1000"] = stat_dict(jac1000[0]), stat_dict(jac1000[1])
    stats["prob_std1"] = stat_dict(prob_std1)
    stats["prob_std2"] = stat_dict(prob_std2)
    stats["prob_std3"] = stat_dict(prob_std3)

    outfile = get_res_filepath(output)
    json.dump(dict(stats=stats, details=res), open(outfile, 'w'), indent=2)
    logging.info("Job finished, results saved at '{}'".format(outfile))
