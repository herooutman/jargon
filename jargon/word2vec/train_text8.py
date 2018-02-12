#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import gensim
# import numpy as np

from monster.misc import get_res_filepath

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def train(args):
    # log_config = dict(name=__file__, console_verbosity=logging.INFO)
    # logger = init_log(**log_config)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    text8 = gensim.models.word2vec.Text8Corpus("/u/kanyuan/text8/text8_nonstop", max_sentence_length=10000)

    sentences = list(text8)
    outfile = get_res_filepath(fn="text8_nonstop.model_1")
    #  -cbow 0 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
    model = gensim.models.Word2Vec(
        sentences,
        workers=20,
        window=10,
        negative=25,
        sg=1,
        size=200,
        sample=0.0001,
        iter=15,
        compute_loss=True)
    model.save(outfile)
