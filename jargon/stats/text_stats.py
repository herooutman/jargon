#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import sqlite3
import json
import pickle
from collections import Counter
from monster.misc import get_res_filepath
from monster.log import init_log

PREPROCESSED_DIR = os.path.abspath(os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


class TextStats(object):
    def __init__(self, data_type, out_dir, logger):
        self.logger = logger
        self.type = data_type
        dbfile = os.path.join(PREPROCESSED_DIR,
                              "{}_preprocessed.db".format(self.type))
        self.conn = sqlite3.connect(dbfile)
        self.cursor = self.conn.cursor()
        self.counter = Counter()
        self.out_dir = out_dir

    def analyze(self):
        sql = "SELECT texts FROM {};".format(self.type)
        self.cursor.execute(sql)
        for row in self.cursor.fetchall():
            text = row[0]
            self.counter.update(text.split())
        json_file = os.path.join(self.out_dir, "{}_texts_stats.json".format(self.type))
        pickle_file = os.path.join(self.out_dir, "{}_texts_stats.pickle".format(self.type))
        with open(json_file, "w") as fd:
            json.dump(dict(self.counter), fd, indent=2, sort_keys=True)
        with open(pickle_file, "wb") as fd:
            pickle.dump(dict(self.counter), fd)
        self.logger.info("FINISHED! Results saved at '{}' and '{}'".format(
            json_file, pickle_file))
        self.conn.close()


def get_text_stats(args):

    parser = argparse.ArgumentParser(description="args for preprocess")
    parser.add_argument("data_type", choices=["reddit", "hackforums", "darkode", "nulled"], type=str)
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

    TextStats(
        data_type=options.data_type, out_dir=out_dir, logger=logger).analyze()
