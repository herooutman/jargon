#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import os
import logging
import sqlite3
import json
import pickle
from collections import Counter
from monster.misc import get_res_filepath
from monster.log import init_log

PREPROCESSED_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


class Coverage(object):
    darklists = ["darkode", "hackforums", "nulled"]
    whitelists = ["wiki", "reddit", "cnet"]
    frequent_bar = 10

    def __init__(self, out_dir, logger):
        self.logger = logger
        self.out_dir = out_dir
        self.vocab_dir = get_res_filepath()
        self.dark_vocabs = dict()
        self.white_vocabs = dict()

    def analyze(self):
        dark_combined = Counter()
        for d in Coverage.darklists:
            dark_vocab_file = os.path.join(self.vocab_dir,
                                           "{}_texts_stats.pickle".format(d))
            if os.path.isfile(dark_vocab_file):
                self.dark_vocabs[d] = pickle.load(open(dark_vocab_file, "rb"))
                dark_combined.update(self.dark_vocabs[d])
                Coverage.stats(d, self.dark_vocabs[d])
        dark_frequent_words = Coverage.stats("Dark Combined", dark_combined)

        white_combined = Counter()
        for w in Coverage.whitelists:
            white_vocab_file = os.path.join(self.vocab_dir,
                                            "{}_texts_stats.pickle".format(w))
            if os.path.isfile(white_vocab_file):
                self.white_vocabs[w] = pickle.load(
                    open(white_vocab_file, "rb"))
                white_combined.update(self.white_vocabs[w])
                Coverage.stats(w, self.white_vocabs[w])
        white_frequent_words = Coverage.stats("White Combined", white_combined)

        Coverage.get_coverage(white=white_frequent_words, dark=dark_frequent_words)

    @staticmethod
    def stats(label, vocab):
        unique_words_ct = len(vocab)
        total_words_ct = sum(vocab.values())
        dist = Counter()
        for _, count in vocab.items():
            dist[count] += 1
        frequent_words = {
            word: count
            for word, count in vocab.items() if count > Coverage.frequent_bar
        }
        frequent_words_ct = len(frequent_words)
        total_frequent_words_ct = sum(frequent_words.values())

        summary = [""]
        summary.append("====== Summary: {} =====".format(label))
        summary.append("Total words count: {}".format(total_words_ct))
        summary.append("Unique words count: {}".format(unique_words_ct))
        summary.append(
            "Frequent words (unique) count: {}".format(frequent_words_ct))
        summary.append("Frequent words coverage: {:.2%}({}/{})".format(
            (total_frequent_words_ct / total_words_ct
             ), total_frequent_words_ct, total_words_ct))
        remain_words_ct = unique_words_ct
        summary.append("")
        summary.append("Low frequency words distribution")
        summary.append("Freq\tCount\tRemain\tSamples")
        for i in range(1, Coverage.frequent_bar + 1):
            remain_words_ct -= dist[i]
            samples = random.sample(
                [word for word in vocab if vocab[word] == i], 5)
            summary.append("{}\t{}\t{}\t{}".format(i, dist[i], remain_words_ct,
                                                   ", ".join(samples)))
        summary.append("")
        summary = "\n".join(summary)
        print(summary)
        return frequent_words

    @staticmethod
    def get_coverage(white, dark):
        white_set = white.keys()
        dark_set = dark.keys()
        common_set = white_set & dark_set
        coverage = len(common_set) / len(dark_set)
        dark_total = sum(dark.values())
        common_total = sum([dark[x] for x in common_set])
        print("common unique words coverage: {:.2%} ({}/{})".format(coverage, len(common_set), len(dark_set)))
        print("common words coverage: {:.2%} ({}/{})".format((common_total / dark_total), common_total, dark_total))

        missed_words = {x: dark[x] for x in (dark_set - white_set)}
        outfile = get_res_filepath("missed_words.json")
        with open(outfile, 'w') as fd:
            json.dump(missed_words, fd, indent=2)
        print(outfile)
        common_words = {x: dark[x] for x in common_set}
        outfile = get_res_filepath("common_words.json")
        with open(outfile, 'w') as fd:
            json.dump(common_words, fd, indent=2)
        print(outfile)


def coverage(args):

    parser = argparse.ArgumentParser(description="args for coverage")
    parser.add_argument(
        "-o", "--out_dir", help="output dir", type=str, default="")
    parser.add_argument(
        "-d", "--debug", help="debug mode", action="store_true", default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose mode",
        action="store_true",
        default=False)

    options = parser.parse_args(args)
    log_config = dict(name=__file__, debug=options.debug)
    out_dir = get_res_filepath(options.out_dir)
    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    Coverage(out_dir=out_dir, logger=logger).analyze()
