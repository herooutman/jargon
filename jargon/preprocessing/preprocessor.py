import os
import time
import sqlite3
import logging
import traceback
import re
import argparse

# from queue import Queue
from abc import ABC, abstractmethod
from gensim.utils import tokenize
from nltk.corpus import stopwords
from multiprocessing import Pool, Queue, Process, current_process, Value
from queue import Empty
from threading import Thread
from enum import Enum
from bs4 import BeautifulSoup

from monster.log import init_log
from monster.misc import pretty_time_delta, format_time
from monster.atomic import AtomicCounter
from monster.misc import get_res_filepath

en_stopwords = stopwords.words('english')

pattern_url = re.compile(
    r"\b([a-z]{3,7}://)((([a-zA-Z*]|[a-zA-Z*][*a-zA-Z0-9\-]*[a-zA-Z0-9])\.)+([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(\/([^\s\[\]\(\)\{\}'\"]*)[a-zA-Z0-9/=])?|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\/[^\s\[\]\(\)\{\}'\"]*[a-zA-Z0-9/=])?)\/?(\b|\s)", flags=re.I
)

pattern_email = re.compile(
    r"\b[a-zA-Z0-9][a-zA-Z0-9._-]+\s*@\s*(([a-zA-Z][a-zA-Z0-9\-]*[a-zA-Z0-9]|[a-zA-Z])\.)+([A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z])\b",
    flags=re.I)

pattern_hash = re.compile(
    r"\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b", flags=re.I)

pattern_likely_url = re.compile(
    r"\b(([a-zA-Z*]|[a-zA-Z*][*a-zA-Z0-9\-]*[a-zA-Z0-9])\.)+([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(\/([^\s\[\]\(\)\{\}'\"]*)[a-zA-Z0-9/=])?\/?(\b|\s)"
)


class Preprocessor(ABC):
    name = "abstract"

    class Command(Enum):
        QUIT = 1

    def __init__(self, in_data, outdir, logger, workers=20, pool_mode=True):
        self.in_data = in_data
        self.logger = logger
        self.outdir = outdir
        self.queue = Queue(maxsize=100)
        self.tasks = Queue(maxsize=1000)
        self.all_loaded = Value('i', False)
        self.write_progress = 0
        self.total_tasks = 0
        self.process_progress = AtomicCounter()
        self.db_file = os.path.join(
            self.outdir, "{}_preprocessed.db.tmp".format(self.__class__.name))
        self.db_writer = None
        self.meta_thread = Thread(target=self._collect_meta, name="meta thread")
        self.writer_thread = Thread(target=self._db_writer_target, name="writer thread")
        self.pool_mode = pool_mode
        self.workers = [
            Process(target=self._worker_process_target, args=(self.tasks,)) for _ in range(workers)
            if not self.pool_mode
        ]
        self.pool = Pool(workers) if self.pool_mode else None
        self.logger.info("{} initialized with {} workers...".format(
            self.__class__.__name__, len(self.workers)))

    def _collect_meta(self):
        self.logger.info("collecting metadata of current job")
        if os.path.isdir(self.in_data):
            self._collect_meta_from_fs()
        else:
            self._collect_meta_from_db(self.in_data)
        return

    # def _collect_meta_from_fs(self):
    #     for root, dirs, filenames in os.walk(self.in_data):
    #         if root.startswith(
    #                 "/home/kanyuan/tools/ForumCrawler/spider_cache/reddit_spider/others/"
    #         ):
    #             continue
    #         for fn in [f for f in filenames if f == "response_body"]:
    #             path = os.path.join(root, fn)
    #             task = dict(path=path)
    #             self.tasks.put(task)
    #     self.logger.info("collected {} tasks".format(self.tasks.qsize()))

    def start(self):
        if self.pool_mode:
            self.start_pool()
        else:
            self.start_queue()
        return

    def start_pool(self):
        self.logger.info(
            "start pre-processing `{dir}` using `{processor}`".format(
                processor=self.__class__.__name__, dir=self.in_data))
        # self.meta_thread.start()
        # self.writer_thread.start()
        #
        # self.pool.apply(self._process_pool_target, (self.tasks))
        #
        # self.pool.close()
        # self.pool.join()
        # self.queue.join()
        #
        # # shutdown db_thread
        # self.queue.put(Preprocessor.Command.QUIT)
        # self.queue.join()
        # self.logger.info("Perprocessing finished, output saved at '{}'...".
        #                  format(self.db_file))

    def start_queue(self):
        self.logger.info(
            "start pre-processing `{dir}` using `{processor}`".format(
                processor=self.__class__.__name__, dir=self.in_data))
        self.meta_thread.start()
        self.writer_thread.start()

        for worker in self.workers:
            worker.start()
        for worker in self.workers:
            worker.join()

        self.logger.info("all workers joined")

        self.queue.put(Preprocessor.Command.QUIT)
        self.writer_thread.join()

        self.logger.info("Perprocessing finished, output saved at '{}'...".
                         format(self.db_file))
        return

    @staticmethod
    def _extract_comment(raw):
        soup = BeautifulSoup(raw, 'lxml')
        # extract code section
        for x in soup.find_all('code'):
            x.extract()
        return soup.get_text(" ")

    def _db_writer_target(self):
        job_start_time = time.time()
        self.db_writer = sqlite3.connect(self.db_file)
        self._create_table()
        while True:
            item = self.queue.get()

            if isinstance(item, Preprocessor.Command):
                if item == Preprocessor.Command.QUIT:
                    break
            else:
                try:
                    self._insert_data(item)
                except sqlite3.OperationalError as e:
                    self.logger.error(
                        "SQLERROR ({}): failed to insert data of '{}' into database '{}'".
                        format(e, str(item)[:100], self.db_file))
                    break
                self.write_progress += 1
                if self.write_progress % 1000 == 0:
                    delta = time.time() - job_start_time
                    rate = self.write_progress / delta
                    eta = self.total_tasks / rate - delta
                    percent = (self.write_progress / self.total_tasks) if self.all_loaded.value else float('nan')
                    self.logger.info(
                        """wrote {prog} articles ({percent:.02%}) in {delta}, processing rate is {rate:.01f} articles/sec, ETA is {eta}""".
                        format(
                            prog=self.write_progress,
                            percent=percent,
                            delta=pretty_time_delta(delta),
                            rate=rate,
                            eta=pretty_time_delta(eta)
                            if self.all_loaded.value else "unknown"))
        self.db_writer.close()
        job_end_time = time.time()
        self.logger.info("job finished at {}, time usage: {}".format(
            format_time(), pretty_time_delta(job_end_time - job_start_time)))
        return

    @staticmethod
    def _extract_texts(content):
        """
        remove all '\n'
        gensim's tokenizer and to lowercase
        remove stop words
        remove infrequent words
        """
        try:
            title = content.get("title", "")
            comments = content.get("comments", [])
            authors = content.get("authors", set())

            comments = map(lambda x: x.replace("\n", " "), comments)
            raw_texts = "{} {}".format(title, "\n".join(comments))
            raw_texts = raw_texts.lower()

            raw_texts = pattern_email.sub("EMAIL__TOKEN ", raw_texts)
            raw_texts = pattern_url.sub("URL__TOKEN ", raw_texts)
            raw_texts = pattern_hash.sub("HASH__TOKEN ", raw_texts)
            # raw_texts = pattern_likely_url.sub("HASH__TOKEN", raw_texts)

            # tokenize
            # from nltk.tokenize import RegexpTokenizer
            # tokenizer = RegexpTokenizer('[a-zA-Z][a-zA-Z0-9]*')
            tokens = list(tokenize(raw_texts, deacc=True))
            # TODO lemmatize

            # remove authors
            tokens = [("AUTHOR__TOKEN" if t in authors else t) for t in tokens]
            # stopwords
            tokens = [t.strip('_ ') for t in tokens if t not in en_stopwords]
            tokens = [t for t in tokens if t]
            # TODO infrequent words
            texts = " ".join(tokens)
            return texts
        except Exception as e:
            traceback.print_exc()

    def _process_pool_target(self, tasks):
        while True:
            try:
                task = tasks.get(block=not self.all_loaded.value)
            except Empty:
                break

            try:
                content = self._parse_raw(task)
            except Exception:
                self.logger.error(traceback.format_exc())
                content = dict()
            raw_comments = content.get("raw_comments", [])
            comments = [self._extract_comment(x) for x in raw_comments]
            content["comments"] = comments
            pure_texts = self._extract_texts(content)
            self.queue.put((content, pure_texts))

            progress = self.process_progress.increment()
            if progress % 100 == 0:
                self.logger.debug(
                    """processed {prog} articles, current queue size: {qsize}....""".
                    format(prog=progress, qsize=self.tasks.qsize()))

    def _worker_process_target(self, tasks):
        self.logger.info("<{}> started...".format(current_process().name))
        while True:
            try:
                task = tasks.get(block=not self.all_loaded.value)
            except Empty:
                self.logger.warning("queue empty")
                break

            # self.logger.info(task)
            # if isinstance(task, Preprocessor.Command):
            #     if task == Preprocessor.Command.QUIT:
            #         self.tasks.task_done()
            #         break
            try:
                content = self._parse_raw(task, self.logger)
            except Exception:
                traceback.print_stack()
                continue
            raw_comments = content.get("raw_comments", [])
            comments = [self._extract_comment(x) for x in raw_comments]
            content["comments"] = comments
            pure_texts = self._extract_texts(content)
            self.queue.put((content, pure_texts))

            progress = self.process_progress.increment()
            if progress % 100 == 0:
                try:
                    qsz = tasks.qsize()
                except NotImplementedError:
                    qsz = float('nan')
                self.logger.debug(
                    """processed {prog} articles, current queue size: {qsize}....""".
                    format(prog=progress, qsize=qsz))
        self.logger.info("<{}> quited...".format(current_process().name))
        return

    @staticmethod
    @abstractmethod
    def _parse_raw(task):
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

    @abstractmethod
    def _collect_meta_from_fs(self):
        pass


def preprocess(args):
    parser = argparse.ArgumentParser(description="args for preprocess")
    parser.add_argument("data_type", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument(
        "-o", "--out_dir", help="output dir", type=str, default="")
    parser.add_argument(
        "-w", "--workers", help="number of workers", type=int, default=20)
    parser.add_argument(
        "-d", "--debug", help="debug mode", action="store_true", default=False)
    parser.add_argument(
        "-q", "--queue", help="queue mode", action="store_true", default=False)
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

    processor = get_processor(options.data_type, options.input, out_dir,
                              logger, options.workers, options.queue)
    processor.start()


def get_processor(data_type, in_data, outdir, logger, workers, queue_mode=True):
    pool_mode = not queue_mode
    if data_type in ('reddit', 'subreddit', 'techreddit'):
        from .reddit_preprocessor import RedditPreprocessor
        return RedditPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            reddit_type=data_type,
            workers=workers,
            pool_mode=pool_mode)
    elif data_type == 'nulled':
        from .nulled_preprocessor import NulledPreprocessor
        return NulledPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers,
            pool_mode=pool_mode)
    elif data_type == 'darkode':
        from .darkode_preprocessor import DarkodePreprocessor
        return DarkodePreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers,
            pool_mode=pool_mode)
    elif data_type == 'hackforums':
        from .hackforums_preprocessor import HackforumsPreprocessor
        return HackforumsPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers,
            pool_mode=pool_mode)
    elif data_type == 'cnet':
        from .cnet_preprocessor import CnetPreprocessor
        return CnetPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers,
            pool_mode=pool_mode)
    elif data_type == 'silkroad':
        from .silkroad_preprocessor import SilkroadPreprocessor
        return SilkroadPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers,
            pool_mode=False)
    else:
        raise Exception("unknown data type")
