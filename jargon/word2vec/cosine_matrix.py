# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import argparse
# import os
# import logging
# import time
# import math
# import numpy as np
#
# from monster.atomic import AtomicCounter
# from itertools import product
# from multiprocessing.pool import ThreadPool
# from gensim.models import Word2Vec
# from monster.misc import get_res_filepath
# from ..utils.math import cosine
#
# DATA_DIR = os.path.abspath(
#     os.path.join(get_res_filepath(), os.pardir, "preprocessing"))
#
#
# def load_cmodel(model_fn):
#     model = dict()
#     with open(model_fn) as fd:
#         rows, columns = map(int, fd.readline().split())
#         for idx, line in enumerate(fd):
#             line = line.strip()
#             fields = line.split()
#             if (len(fields) != columns + 1):
#                 logging.error("malformatted model file")
#             else:
#                 word = fields[0]
#                 vector = np.array([float(x) for x in fields[1:]])
#                 model[word] = vector
#     return model
#
#
# def load_gensim_model(model_fn):
#     model = dict()
#     gensim_model = Word2Vec.load(model_fn)
#     syn0 = gensim_model.wv.syn0
#     for idx, vector in enumerate(syn0):
#         word = gensim_model.wv.index2word[idx]
#         model[word] = vector
#     return model
#
#
# def load_model(model_fn):
#     try:
#         return load_cmodel(model_fn)
#     except UnicodeDecodeError:
#         return load_gensim_model(model_fn)
#
#
# def cosine_matrix(args):
#     logging.basicConfig(
#         format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
#     parser = argparse.ArgumentParser(description="args for prediction")
#     parser.add_argument(
#         "-g", "--good", help="good model, from c++ or gensim", type=str)
#     parser.add_argument(
#         "-b", "--bad", help="bad model, from c++ or gensim", type=str)
#     parser.add_argument(
#         "-t", "--thread", help="number of threads", type=int, default=1)
#     parser.add_argument(
#         "-o",
#         "--output",
#         help="output filename",
#         type=str,
#         default="cosine_matrix.npy")
#
#     options = options = parser.parse_args(args)
#
#     if options.good and os.path.isfile(options.good):
#         good = load_model(options.good)
#         logging.info("good model '{}' loaded".format(options.good))
#     else:
#         logging.error(
#             "Error: good model file '{}' not found".format(options.good))
#         return 1
#
#     if options.bad and os.path.isfile(options.bad):
#         bad = load_model(options.bad)
#         logging.info("bad model '{}' loaded".format(options.bad))
#     else:
#         logging.error(
#             "Error: bad model file '{}' not found".format(options.bad))
#         return 1
#     if options.thread < 0:
#         logging.error("Error: number of threads must be larger than 0")
#     cosine_matrix_impl(good, bad, options.output, options.thread)
#
#
# def compute_thread(g, b, progress):
#     print(g)
#     gi, gv = g
#     bi, bv = b
#     c = cosine(gv, bv)
#     pv = progress.increment()
#     if pv % 10 == 0:
#         current_time = time.time()
#         rate = pv / (current_time - start_time) / 70000
#         logging.info(
#             """processed {prog} words, processing rate is {rate:.02f} word/sec....""".
#             format(prog=pv, rate=rate))
#
#
# def cosine_matrix_impl(good, bad, output, workers):
#     global start_time
#     if good.keys() != bad.keys():
#         logging.error("unmatched models")
#         return
#
#     index2word = list()
#     vocab = dict()
#     goodlist = list()
#     badlist = list()
#     for word in good.keys():
#         vocab[word] = len(index2word)
#         index2word.append(word)
#         goodlist.append(good[word])
#         badlist.append(bad[word])
#
#     matrix = np.empty((len(vocab), len(vocab)))
#     matrix.fill(10)
#     progress = AtomicCounter()
#     logging.info("start calculating")
#     pool = ThreadPool(processes=workers)
#     start_time = time.time()
#     pool.starmap(compute_thread, product(enumerate(goodlist), enumerate(badlist), [progress]))
#
#     # for gi, gv in enumerate(goodlist):
#     #     for bi, bv in enumerate(badlist):
#     #         pool.apply_async(
#     #             compute_thread,
#     #             args=(gi, gv, bi, bv, progress))
#     logging.info("finish depatching")
#     pool.close()
#     pool.join()
#     np.save(output, matrix)
#     logging.info("results saved at '{}'".format(output))
