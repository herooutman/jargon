import json
import os
import time
import sqlite3
import logging
import traceback
import re
import argparse
import hashlib

from collections import defaultdict
from nltk.tokenize import sent_tokenize
# from queue import Queue
from abc import ABC, abstractmethod
from gensim.utils import tokenize
from nltk.corpus import stopwords
from multiprocessing import Queue, Process, current_process, Value
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
    r"\b([a-z]{3,7}://)((([a-zA-Z*]|[a-zA-Z*][*a-zA-Z0-9\-]*[a-zA-Z0-9])\.)+([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(\/([^\s\[\]\(\)\{\}'\"]*)[a-zA-Z0-9/=])?|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\/[^\s\[\]\(\)\{\}'\"]*[a-zA-Z0-9/=])?)\/?(\b|\s)"
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

    def __init__(self, in_data, outdir, logger, workers=20):
        self.in_data = in_data
        self.logger = logger
        self.outdir = outdir
        self.queue = Queue(maxsize=100)
        self.tasks = Queue(maxsize=1000)
        self.all_loaded = Value('i', False)
        self.write_progress = 0
        self.total_tasks = 0
        self.process_progress = AtomicCounter()
        self.db_writer = None
        self.out_file = os.path.join(
            self.outdir, "{}_dated.json".format(self.__class__.name))
        self.meta_thread = Thread(target=self._collect_meta, name="meta thread")
        self.writer_thread = Thread(target=self._db_writer_target, name="writer thread")
        self.workers = [
            Process(target=self._worker_process_target, args=(self.tasks,)) for _ in range(workers)
        ]
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
        self.start_queue()

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
                         format(self.out_file))
        return

    @staticmethod
    def _extract_comment(raw):
        soup = BeautifulSoup(raw, 'lxml')
        # extract code section
        for x in soup.find_all('code'):
            x.extract()
        return soup.get_text(" ")

    def _db_writer_target(self):
        res = defaultdict(list)
        hash_dict = dict()
        job_start_time = time.time()
        while True:
            item = self.queue.get()

            if isinstance(item, Preprocessor.Command):
                if item == Preprocessor.Command.QUIT:
                    break
            else:
                more_res, more_hash_dict = item
                hash_dict.update(more_hash_dict)
                for token in more_res:
                    res[token].extend(more_res[token])

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
        with open(self.out_file, 'w') as ofd:
            json.dump([res, hash_dict], ofd)
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
        res = defaultdict(list)
        all_tokens = set()
        m = hashlib.md5()
        try:
            title = content.get("title", "")
            comments = content.get("comments", [])
            comments = [title] + comments
            dates = content.get("dates", [])

            m.update(json.dumps(content).encode('utf-8'))
            thread_hash = m.hexdigest()

            hash_dict = {thread_hash: content}
            threads_meta = dict(start_date=min(dates), end_date=max(dates), thread_hash=thread_hash)

            if dates:
                dates = [dates[0]] + dates
            else:
                return res

            for idx, comment in enumerate(comments):
                comment = comment.strip()
                date = dates[idx]
                if idx <= 1 or len(comment) >= 20:
                    comment = comment.lower()
                    comment = pattern_email.sub("EMAIL__TOKEN ", comment)
                    comment = pattern_url.sub("URL__TOKEN ", comment)
                    comment = pattern_hash.sub("HASH__TOKEN ", comment)

                    for para in comment.splitlines():
                        for sent in sent_tokenize(para):
                            tokens = tokenize(sent, deacc=True)
                            tokens = filter(
                                lambda x: (not x.isdigit()) and len(x) > 1,
                                tokens)
                            tokens = [
                                t.strip('-_ ') for t in tokens
                                if t not in en_stopwords
                            ]
                            all_tokens.update({t for t in tokens if t})
            if len(all_tokens) > 3:
                for token in all_tokens:
                    res[token].append(threads_meta)
            return res, hash_dict
        except Exception as e:
            traceback.print_exc()


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
                if not content:
                    continue
            except Exception:
                traceback.print_stack()
                continue
            raw_comments = content.get("raw_comments", [])
            comments = [self._extract_comment(x) for x in raw_comments]
            content['authors'] = list(content.get('authors', list()))
            content['comments'] = comments
            results = self._extract_texts(content)
            if results:
                self.queue.put(results)

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
        "-v",
        "--verbose",
        help="verbose mode",
        action="store_true",
        default=False)

    options = parser.parse_args(args)
    log_config = dict(name=__file__, debug=options.debug)
    out_dir = get_res_filepath(folder=options.out_dir)
    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    processor = get_processor(options.data_type, options.input, out_dir,
                              logger, options.workers)
    processor.start()


def get_processor(data_type, in_data, outdir, logger, workers):
    if data_type in ('reddit', 'subreddit', 'techreddit'):
        from .reddit_preprocessor import RedditPreprocessor
        return RedditPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            reddit_type=data_type,
            workers=workers)
    elif data_type == 'nulled':
        from .nulled_preprocessor import NulledPreprocessor
        return NulledPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers)
    elif data_type == 'darkode':
        from .darkode_preprocessor import DarkodePreprocessor
        return DarkodePreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers)
    elif data_type == 'hackforums':
        from .hackforums_preprocessor import HackforumsPreprocessor
        return HackforumsPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers)
    elif data_type == 'cnet':
        from .cnet_preprocessor import CnetPreprocessor
        return CnetPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers)
    elif data_type == 'silkroad':
        from .silkroad_preprocessor import SilkroadPreprocessor
        return SilkroadPreprocessor(
            in_data=in_data,
            outdir=outdir,
            logger=logger,
            workers=workers)
    else:
        raise Exception("unknown data type")
