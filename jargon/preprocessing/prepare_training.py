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

all_dark_forums = ["darkode", "hackforums", "nulled", "silkroad"]
all_white_forums = ["reddit", "cnet", "wiki"]

frequent_bar = 10


class TrainingPrepare(object):
    def __init__(self, in_dir, out_dir, logger, dark, white, wiki, workers):
        self.logger = logger
        self.out_dir = out_dir
        self.db_dir = in_dir
        self.dark_selections = dark
        self.white_selections = white
        self.wiki = wiki
        self.logger.info("Init finished")
        self.pool = ThreadPool(processes=workers)
        self.logger.info(
            "Prepare training data from dark_forums ({}) and white forums ({})".
            format("/".join(self.dark_selections), "/".join(
                self.white_selections + (["wiki"] if self.wiki else []))))

    def go(self):
        dark_threads = dict()
        dark_vocab = Counter()
        for forum in self.dark_selections:
            dark_threads[forum] = self.get_threads(forum, dark_vocab)
        self.logger.info("dark_vocab: {}".format(len(dark_vocab)))
        # dark_lines = self.filter_infrequent(dark_lines, "dark")

        white_threads = dict()
        white_vocab = Counter()
        for forum in self.white_selections:
            white_threads[forum] = self.get_threads(forum, white_vocab)
        self.logger.info("white_vocab: {}".format(len(white_vocab)))
        if self.wiki:
            white_threads['wiki'] = self.get_wiki_threads(forum)
        # white_lines = self.filter_infrequent(white_lines, "white")
        self.pool.close()

        common_vocab = dark_vocab.keys() & white_vocab.keys()
        common_vocab_plus = common_vocab | {
            k
            for k in dark_vocab if dark_vocab[k] >= 200
        }
        self.logger.info("common vocab: {}, plus: {}".format(
            len(common_vocab), len(common_vocab_plus)))
        # writing data
        dark_out = os.path.join(self.out_dir, "dark_texts.txt")
        white_out = os.path.join(self.out_dir, "white_texts.txt")

        self.write_data(dark_out, dark_threads, common_vocab_plus)
        self.write_data(white_out, white_threads, common_vocab_plus)
        self.logger.info(
            "Dark text file saved at '{}'\nWhite text file saved at '{}'".
            format(dark_out, white_out))

    def write_data(self, out_file, threads, common_vocab):
        fd = open(out_file, "w")
        ignored = 0
        processed = 0
        for forum, thread_list in threads.items():
            subfile = os.path.join(self.out_dir, "{}.txt".format(forum))
            subfd = open(subfile, "w")
            for thread in thread_list:
                thread_texts = []
                for para in thread:
                    for sent in para:
                        sent_vocab = set(sent)
                        processed += 1
                        if sent_vocab.issubset(common_vocab):
                            thread_texts.append(" ".join(sent))
                        else:
                            ignored += 1
                if thread_texts:
                    fd.write("{}\n".format(" ".join(thread_texts)))
                    subfd.write("{}\n".format(" ".join(thread_texts)))
            subfd.flush()
            subfd.close()
        fd.flush()
        fd.close()
        self.logger.info("processed sentences: {}, ignored sentences: {}".
                         format(processed, ignored))

    def get_threads(self, forum, vocab):
        batch_size = 1000
        threads = list()

        sql = "SELECT title, comments, authors FROM {}_fts;".format(forum)

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
                    title, comments, authors = row
                    comments = json.loads(comments)
                    authors = json.loads(authors)
                    tasks.append(([title] + comments, authors))

                results = self.pool.starmap(self.form_thread, tasks, 100)
                for res in results:
                    if res:
                        threads.append(res[0])
                        vocab.update(res[1])
                batch_ct += 1
                batch = cursor.fetchmany(batch_size)
                tasks.clear()
        except Exception:
            traceback.print_exc()

        self.logger.info("result length: {}".format(len(threads)))
        db.close()
        return threads

    @staticmethod
    def form_thread(comments, authors):
        vocab = Counter()
        try:
            thread = list()
            for idx, comment in enumerate(comments):
                comment = comment.strip()
                if idx <= 1 or len(comment) >= 20:
                    comment = comment.lower()
                    comment = pattern_email.sub("EMAIL__TOKEN ", comment)
                    comment = pattern_url.sub("URL__TOKEN ", comment)
                    comment = pattern_hash.sub("HASH__TOKEN ", comment)
                    sentences = list()

                    for para in comment.splitlines():
                        for sent in sent_tokenize(para):
                            tokens = tokenize(sent, deacc=True)
                            tokens = filter(
                                lambda x: (not x.isdigit()) and len(x) > 1,
                                tokens)
                            # tokens = [("AUTHOR__TOKEN" if t in authors else t)
                            #           for t in tokens]
                            tokens = [
                                t.strip('-_ ') for t in tokens
                                if t not in en_stopwords
                            ]
                            tokens = [t for t in tokens if t]
                            if tokens:
                                sentences.append(tokens)
                                vocab.update(tokens)

                    thread.append(sentences)
                else:
                    pass

            return thread, vocab
        except Exception as e:
            traceback.print_exc()

    def get_wiki_threads(self):
        wiki_file = os.path.join(self.db_dir, "wiki_en.textonly.db")
        with open(wiki_file) as fd:
            lines = fd.readlines()
            # for line in lines:
            #     print(line)
        return lines

    def filter_infrequent(self, lines, label):
        c = Counter()
        for idx, line in enumerate(lines):
            c.update(line.split())
            if idx % 10000 == 0 and idx > 0:
                self.logger.info("Read {}({:02%}) {} lines.".format(
                    idx, idx / len(lines), label))
        for idx, line in enumerate(lines):
            if idx % 10000 == 0 and idx > 0:
                self.logger.info("Filtered {}({:02%}) {} lines.".format(
                    idx, idx / len(lines), label))
            yield " ".join([w for w in line.split() if c[w] > frequent_bar])


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
        choices=all_dark_forums + all_white_forums,
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

    dark_selections = []
    white_selections = []
    wiki = False

    for choice in options.forums:
        if choice == "wiki":
            wiki = True
        elif choice in all_dark_forums:
            dark_selections.append(choice)
        elif choice in all_white_forums:
            white_selections.append(choice)
    log_config = dict(name=__file__, debug=options.debug)
    out_dir = get_res_filepath(folder=options.out_dir)
    in_dir = os.path.join(PREPROCESSED_DIR, options.in_dir)
    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    TrainingPrepare(
        in_dir=in_dir,
        out_dir=out_dir,
        logger=logger,
        dark=dark_selections,
        white=white_selections,
        wiki=wiki,
        workers=options.workers).go()
