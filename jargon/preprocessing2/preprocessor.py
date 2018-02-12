import os
import time
import sqlite3
import json
import logging
import traceback
import argparse

from queue import Queue
from abc import ABC, abstractmethod
from gensim.utils import tokenize
from nltk.corpus import stopwords
from multiprocessing.pool import ThreadPool
from threading import Thread
from enum import Enum
from bs4 import BeautifulSoup

from monster.log import init_log
from monster.misc import pretty_time_delta, format_time

from ..utils.misc import get_res_filepath

en_stopwords = stopwords.words('english')


class Preprocessor(ABC):
    name = "general"

    class Command(Enum):
        QUIT = 1

    def __init__(self, in_data, outdir, logger, workers=20):
        self.in_data = in_data
        self.logger = logger
        self.outdir = outdir
        self.progress = 0
        self.queue = Queue()
        self.tasks = list()
        self.db_file = os.path.join(self.outdir, "{}_preprocessed.db".format(self.__class__.name))
        self.db_conn = None
        self.db_thread = Thread(target=self._db_thread_target)
        self.pool = ThreadPool(processes=workers)

    def _collect_meta(self):
        self.logger.info("collecting metadata of current job")
        if os.path.isdir(self.in_data):
            self._collect_meta_from_fs()
        else:
            self._collect_meta_from_db(self.in_data)

    def _collect_meta_from_fs(self):
        for root, dirs, filenames in os.walk(self.in_data):
            if root.startswith(
                    "/home/kanyuan/tools/ForumCrawler/spider_cache/reddit_spider/others/"
            ):
                continue
            for fn in [f for f in filenames if f == "response_body"]:
                path = os.path.join(root, fn)
                task = dict(path=path)
                self.tasks.append(task)
                if (len(self.tasks) % 1000 == 0):
                    self.logger.info(
                        "scanned {} files".format(len(self.tasks)))
        self.logger.info("collected {} tasks".format(len(self.tasks)))

    def start(self):
        self._collect_meta()
        self.logger.info(
            "start pre-processing `{dir}` using `{processor}`".format(
                processor=self.__class__.__name__, dir=self.in_data))
        self.db_thread.start()

        for task in self.tasks:
            self.pool.apply_async(self._processor_thread_target, (task, ))

        time.sleep(2)
        self.pool.close()
        self.pool.join()
        self.queue.join()

        # shutdown db_thread
        self.queue.put(Preprocessor.Command.QUIT)
        self.queue.join()

    def _extract_comment(self, raw):
        soup = BeautifulSoup(raw, 'lxml')
        return soup.get_text(" ")

    def _processor_thread_target(self, task):
        content = self._parse_raw(task)
        raw_comments = content.get("raw_comments", [])
        comments = [self._extract_comment(x) for x in raw_comments]
        content["comments"] = comments
        pure_texts = self._extract_texts(content)
        self.queue.put((content, pure_texts))

    def _db_thread_target(self):
        job_start_time = time.time()
        self.db_conn = sqlite3.connect(self.db_file)
        self._create_table()
        while True:
            item = self.queue.get()
            if isinstance(item, Preprocessor.Command):
                if item == Preprocessor.Command.QUIT:
                    self.queue.task_done()
                    break
            else:
                try:
                    self._insert_data(item)
                except sqlite3.OperationalError as e:
                    self.logger.error("SQLERROR: failed to insert data of '{}' into database '{}'".format(str(item)[:100], self.db_file))
                    break
                self.progress += 1
                self.queue.task_done()
                if self.progress % 1000 == 0:
                    delta = time.time() - job_start_time
                    rate = self.progress / delta
                    eta = len(self.tasks) / rate - delta
                    percent = self.progress / len(self.tasks)
                    self.logger.info(
                        """processed {prog} articles ({percent:.02%}) in {delta}, processing rate is {rate} articles/sec, ETA is  {eta}""".
                        format(
                            prog=self.progress,
                            percent=percent,
                            delta=pretty_time_delta(delta),
                            rate=rate,
                            eta=pretty_time_delta(eta)))
        self.db_conn.close()
        job_end_time = time.time()
        self.logger.info("job finished at {}, time usage: {}".format(
            format_time(), pretty_time_delta(job_end_time - job_start_time)))

    def _extract_texts(self, content):
        # remove all \n
        # gensim's tokenizer and to lowercase
        # remove stop words
        # remove infrequent words
        try:
            title = content.get("title", "")
            comments = content.get("comments", [])
            comments = map(lambda x: x.replace("\n", " "), comments)
            raw_texts = "{} {}".format(title, "\n".join(comments))
            # tokenize
            # from nltk.tokenize import RegexpTokenizer
            # tokenizer = RegexpTokenizer('[a-zA-Z][a-zA-Z0-9]*')
            tokens = list(tokenize(raw_texts, lower=True, deacc=True))
            # stopwords
            clean_tokens = [t for t in tokens if t not in en_stopwords]
            # infrequent words
            texts = " ".join(clean_tokens)
            return texts
        except Exception as e:
            traceback.print_exc()

    @abstractmethod
    def _parse_raw(self, response_file_path):
        pass

    @abstractmethod
    def _create_table(self):
        pass

    @abstractmethod
    def _insert_data(self, item):
        pass

    @abstractmethod
    def _collect_meta_from_db(self):
        pass


def preprocess(args):
    parser = argparse.ArgumentParser(description="args for preprocess")
    parser.add_argument("data_type", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("-o", "--out_dir", help="output dir", type=str, default="")
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

    processor = get_processor(options.data_type, options.input, out_dir, logger)
    processor.start()


def get_processor(data_type, *args):
    if data_type == 'reddit':
        from .reddit_preprocessor import RedditPreprocessor
        return RedditPreprocessor(*args)
    elif data_type == 'darkode':
        from .darkode_preprocessor import DarkodePreprocessor
        return DarkodePreprocessor(*args)
    elif data_type == 'hackforums':
        from .hackforums_preprocessor import HackforumsPreprocessor
        return HackforumsPreprocessor(*args)
    else:
        raise Exception("unknown data type")
