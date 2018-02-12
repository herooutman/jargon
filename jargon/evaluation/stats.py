#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import glob
import argparse
import os
import logging
import sqlite3
import traceback
import numpy as np
import csv

from gensim.matutils import argsort, unitvec
from gensim.utils import tokenize as tokenize1
from ..utils.misc import tokenize as tokenize2

from collections import defaultdict

from multiprocessing.pool import ThreadPool
from collections import Counter
from nltk.tokenize import sent_tokenize
# from gensim.utils import tokenize

from monster.misc import get_res_filepath
from monster.log import init_log

PREPROCESSED_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))

MODEL_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, os.pardir, "word2vec/jargon"))
tokenize = tokenize2


def load_cmodel(model_fn):
    model = dict()
    with open(model_fn) as fd:
        rows, columns = map(int, fd.readline().split())
        for idx, line in enumerate(fd):
            line = line.strip()
            fields = line.split()
            if (len(fields) != columns + 1):
                logging.error("malformatted model file")
            else:
                word = fields[0]
                vector = np.array([float(x) for x in fields[1:]])
                model[word] = vector
    return model


def load_vocab(vfn):
    res = dict()
    fd = open(vfn)
    for line in fd:
        line = line.strip()
        fields = line.split()
        word, cn, gcn, bcn = fields
        res[word] = dict(cn=int(cn), gcn=int(gcn), bcn=int(bcn))
    return res


def get_vectors(model, words):
    res = list()
    for word in words:
        res.append(model[word])
    return np.array(res)


def normalize(matrix):
    ret = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        ret[i] = unitvec(matrix[i])
    return ret


def closest(vec, matrix_u, word_list=None, topn=40):
    vec_u = unitvec(vec)
    cosines = np.dot(matrix_u, vec_u.T)
    close_index = argsort(cosines, topn=topn, reverse=True)
    if word_list is None:
        res = {i: cosines[i] for i in close_index}
    else:
        res = {word_list[i]: cosines[i] for i in close_index}
    return sorted(res.items(), key=lambda x: x[1], reverse=True)


def get_info(word):
    global gm, bm, vocab, word_list, gvs, bvs, gvs_u, bvs_u

    if word in vocab:
        gcn = vocab[word]['gcn']
        bcn = vocab[word]['bcn']
    else:
        gcn = None
        bcn = None

    if word in word_list:
        idx = word_list.index(word)
        sim = np.dot(gvs_u[idx], bvs_u[idx])
        good_interpretation = [
            (x[0], x[1], vocab[x[0]]['gcn'])
            for x in closest(bm[word], gvs_u, word_list, topn=100)
            if vocab[x[0]]['gcn'] >= 10
        ]
        bad_interpretation = [
            (x[0], x[1], vocab[x[0]]['bcn'])
            for x in closest(bm[word], gvs_u, word_list, topn=100)
            if vocab[x[0]]['bcn'] >= 10
        ]
        normal_interpretation = [
            (x[0], x[1], vocab[x[0]]['gcn'])
            for x in closest(gm[word], gvs_u, word_list, topn=100)
            if vocab[x[0]]['gcn'] >= 10
        ]
    else:
        sim = None
        good_interpretation = []
        bad_interpretation = []
        normal_interpretation = []
    return sim, gcn, bcn, good_interpretation, bad_interpretation, normal_interpretation


def get_candidates(model):
    global gm, bm, vocab, word_list, gvs, bvs, gvs_u, bvs_u
    good_fn = os.path.join(MODEL_DIR, "{}.good.syn0".format(model))
    bad_fn = os.path.join(MODEL_DIR, "{}.bad.syn0".format(model))
    vocab_fn = os.path.join(MODEL_DIR, "{}.vocab".format(model))
    vocab = load_vocab(vocab_fn)

    gm = load_cmodel(good_fn)
    bm = load_cmodel(bad_fn)
    word_list = list(bm.keys())

    gvs = get_vectors(gm, word_list)
    bvs = get_vectors(bm, word_list)
    gvs_u = normalize(gvs)
    bvs_u = normalize(bvs)

    cosines = dict()
    for i in range(len(gvs_u)):
        cosines[word_list[i]] = np.dot(gvs_u[i], bvs_u[i])

    t = [(k, v) for k, v in cosines.items()
         if k in vocab and vocab[k]['gcn'] >= 100 and vocab[k]['bcn'] >= 50]
    rank = 0
    for word, sim in sorted(t, key=lambda x: float(x[1])):
        if sim < .3:
            rank += 1
            good_interpretation = [
                (x[0], x[1], vocab[x[0]]['gcn'])
                for x in closest(bm[word], gvs_u, word_list, topn=100)
                if vocab[x[0]]['gcn'] >= 10
            ]
            bad_interpretation = [
                (x[0], x[1], vocab[x[0]]['bcn'])
                for x in closest(bm[word], gvs_u, word_list, topn=100)
                if vocab[x[0]]['bcn'] >= 10
            ]
            normal_interpretation = [
                (x[0], x[1], vocab[x[0]]['gcn'])
                for x in closest(gm[word], gvs_u, word_list, topn=100)
                if vocab[x[0]]['gcn'] >= 10
            ]
            yield rank, word, sim, good_interpretation, bad_interpretation, normal_interpretation


def find_in_comments(word, comments):
    comments = json.loads(comments)
    for c in comments:
        for line in c.split("\n"):
            tokens = list(tokenize(line.lower(), deacc=True))
            if word in tokens:
                yield line


def prepare_dark(words):
    datasets = ['darkode', 'nulled', 'hackforums']
    return prepare_search(words, datasets)


def prepare_white(words):
    datasets = ['reddit']
    return prepare_search(words, datasets)


def prepare_search(words, datasets):
    res = defaultdict(set)
    for dataset in datasets:
        db_file = os.path.join(PREPROCESSED_DIR,
                               "{}_preprocessed.db".format(dataset))
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # FIXME the join method
        # sql = "CREATE TEMPORARY TABLE patterns (pattern VARCHAR(20))"
        # cursor.execute(sql)
        #
        # sql = "INSERT INTO patterns VALUES (?)"
        # cursor.executemany(sql, [("%{}%".format(word),) for word in words])
        # conn.commit()

        # sql = "SELECT t.comments FROM {} t JOIN patterns p ON (t.comments LIKE p.pattern)".format(
        #     dataset)

        # FIXME the or method
        subs = list()
        for i in range(0, len(words), 2048):
            subs.append(
                "SELECT comments FROM {}_fts WHERE comments MATCH '{}'".format(
                    dataset, " OR ".join(words[i:i + 1000])))
        sql = " UNION ".join(subs)
        cursor.execute(sql)
        for row in cursor.fetchall():
            for word in words:
                if word in row[0]:
                    for sen in find_in_comments(word, row[0]):
                        # sen = sen.replace(",", " ")
                        res[word].add("[{}] {}".format(dataset, sen))
    return res


def label2str(labels):
    forums, types, taggers = list(zip(*labels))
    return "{} {}".format("/".join(set(forums)), "/".join(set(types)))


def stats_impl(annotations, model, out_file, sen):
    out_file = get_res_filepath(fn="{}.csv".format(out_file))
    out_file2 = get_res_filepath(fn="{}_missed.csv".format(out_file))

    logger.info("init finished")

    cands = list(get_candidates(model))
    words = [x[1] for x in cands]
    if sen:
        logger.info("start preparing data")
        db_dark = prepare_dark(words)
        logger.info("dark data preparing finished")
        db_white = prepare_white(words)
        logger.info("white data preparing finished")
    else:
        db_dark = dict()
        db_white = dict()

    csvfd = open(out_file, 'w')
    spamwriter = csv.writer(csvfd)
    for rank, word, sim, good_interpretation, bad_interpretation, normal_interpretation in cands:
        logger.info("processing word '{}''".format(word))
        good_sen = "...... ".join(db_white.get(word, []))
        bad_sen = "...... ".join(db_dark.get(word, []))
        good_interpretation = [x[0] for x in good_interpretation]
        bad_interpretation = [x[0] for x in bad_interpretation]
        normal_interpretation = [x[0] for x in normal_interpretation]
        if word in annotations:
            labeled = True
            label = label2str(annotations[word])
        else:
            labeled = False
            label = ""

        spamwriter.writerow([
            rank, word, good_interpretation, bad_interpretation, bad_sen,
            normal_interpretation, good_sen, labeled, label
        ])

    csvfd.close()

    csvfd = open(out_file2, 'w')
    spamwriter = csv.writer(csvfd)
    spamwriter.writerow([
        "word", "score", "gcn", "bcn", "good_interpretation",
        "bad_interpretation", "bad_sen", "normal_interpretation", "good_sen",
        "labeled", "label"
    ])
    for word in annotations:
        if word not in words:
            sim, gcn, bcn, good_interpretation, bad_interpretation, normal_interpretation = get_info(
                word)
            logger.info("processing missing word '{}''".format(word))
            # good_sen = "...... ".join(db_white.get(word, []))
            # bad_sen = "...... ".join(db_dark.get(word, []))
            good_interpretation = [x[0] for x in good_interpretation]
            bad_interpretation = [x[0] for x in bad_interpretation]
            normal_interpretation = [x[0] for x in normal_interpretation]
            if word in annotations:
                labeled = True
                label = label2str(annotations[word])
            else:
                labeled = False
                label = ""

            spamwriter.writerow([
                word, sim, gcn, bcn, good_interpretation, bad_interpretation,
                bad_sen, normal_interpretation, good_sen, labeled, label
            ])

    csvfd.close()

    logger.info("data saved at '{}' and '{}'".format(out_file, out_file2))


def stats(args):
    global logger
    parser = argparse.ArgumentParser(description="args for parse_annotated")
    parser.add_argument(
        "-a",
        "--annotations",
        help="annotations dir",
        type=str,
        default="annotations.json")
    parser.add_argument(
        "-m", "--model", help="model name", type=str, default="forums.it100")
    parser.add_argument(
        "-o", "--out_file", help="output dir", type=str, default="stats")
    # parser.add_argument(
    #     "-w", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument(
        "-d", "--debug", help="debug mode", action="store_true", default=False)
    parser.add_argument(
        "-s",
        "--sentence",
        help="output sentence",
        action="store_true",
        default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose mode",
        action="store_true",
        default=False)

    options = parser.parse_args(args)

    log_config = dict(name=__file__, debug=options.debug)

    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    annotations = json.load(open(get_res_filepath(fn=options.annotations)))
    stats_impl(
        annotations=annotations,
        model=options.model,
        out_file=options.out_file,
        sen=options.sentence)
