#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
import logging
import sqlite3
import traceback

from multiprocessing.pool import ThreadPool
from collections import Counter
from nltk.tokenize import sent_tokenize
# from gensim.utils import tokenize

from ..utils.misc import tokenize
from monster.misc import get_res_filepath
from monster.log import init_log
from monster.atomic import AtomicCounter
from .preprocessor import pattern_url, pattern_email, pattern_hash, en_stopwords

PREPROCESSED_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))

allforums = ["darkode", "hackforums", "nulled", "silkroad"]

frequent_bar = 10

code_indicators = "(){}[].;\""

class TrainingPrepare(object):
    def __init__(self, in_dir, out_dir, logger, forums, workers):
        self.logger = logger
        self.out_dir = out_dir
        self.db_dir = in_dir
        self.selections = forums
        self.logger.info("Init finished")
        self.pool = ThreadPool(processes=workers)
        self.logger.info(
            "Prepare training data from forums ({})".
            format("/".join(self.selections)))

    def go(self):
        threads = dict()
        for forum in self.selections:
            threads[forum] = self.get_threads(forum)

        self.pool.close()

        # writing data
        thread_out = os.path.join(self.out_dir, "threads.json")

        self.write_data(thread_out, threads)
        self.logger.info(
            "Threads file saved at '{}'".format(thread_out))

    def write_data(self, out_file, threads):
        fd = open(out_file, "w")
        comment_set = set()
        whole_list = list()
        for forum, thread_list in threads.items():
            subfile = os.path.join(self.out_dir, "{}.json".format(forum))
            subfd = open(subfile, "w")
            self.logger.info("writting {} threads to '{}'".format(len(threads), subfile))
            json.dump(thread_list, subfd)
            subfd.close()
            whole_list += thread_list

        json.dump(thread_list, fd)
        fd.close()


    def get_threads(self, forum):
        batch_size = 1000
        threads = list()

        sql = "SELECT title, comments FROM {}_fts;".format(forum)

        db_file = os.path.join(self.db_dir, "{}_preprocessed.db".format(forum))
        db = sqlite3.connect(db_file)
        cursor = db.cursor()
        cursor.execute(sql)
        self.logger.info("start to get threads from {}".format(forum))
        try:
            cursor.execute(sql)
            batch = cursor.fetchmany(batch_size)
            tasks = list()
            batch_ct = 0
            while batch:
                self.logger.info(
                    "Read {} threads.".format(batch_size * batch_ct))
                for row in batch:
                    title, comments = row
                    comments = [title] + json.loads(comments)
                    tasks.append(comments)

                results = self.pool.imap(self.form_thread, tasks, 100)
                for res in results:
                    if len(res) > 100:
                        threads.append(res)

                batch_ct += 1
                batch = cursor.fetchmany(batch_size)
                tasks.clear()
        except Exception:
            traceback.print_exc()

        self.logger.info("result length: {}".format(len(threads)))
        db.close()
        return threads

    @staticmethod
    def form_thread(comments):
        try:
            thread = list()
            for idx, comment in enumerate(comments):
                comment = comment.strip()
                if idx <= 1 or len(comment) >= 20:
                    comment = pattern_email.sub("EMAIL__TOKEN ", comment)
                    comment = pattern_url.sub("URL__TOKEN ", comment)
                    comment = pattern_hash.sub("HASH__TOKEN ", comment)

                    comment_len = len(comment)
                    ctr = Counter(comment)
                    ci_count = sum(ctr[ci] for ci in code_indicators)

                    if comment_len > 200 and ci_count / comment_len > .05:
                        pass
                    else:
                        thread.append(comment)

                else:
                    pass

            return ".\n\n".join(thread)
        except Exception as e:
            traceback.print_exc()


def prepare(args):
    parser = argparse.ArgumentParser(description="args for prepare")
    parser.add_argument(
        "-i", "--in_dir", help="input dir", type=str, default="")
    parser.add_argument(
        "-o", "--out_dir", help="output dir", type=str, default="")
    parser.add_argument(
        "-t", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument(
        "-f",
        '--forums',
        nargs='+',
        required=True,
        choices=allforums,
        help='specifies target forum(s)')
    parser.add_argument(
        "-d", "--debug", help="debug mode", action="store_true", default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose mode",
        action="store_true",
        default=False)

    options = parser.parse_args(args)

    selections = [f for f in options.forums if f in allforums]

    log_config = dict(name=__file__, debug=options.debug)
    out_dir = get_res_filepath(folder=os.path.join('text2data', options.out_dir))
    in_dir = os.path.join(PREPROCESSED_DIR, options.in_dir)
    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    TrainingPrepare(
        in_dir=in_dir,
        out_dir=out_dir,
        logger=logger,
        forums=selections,
        workers=options.workers).go()
