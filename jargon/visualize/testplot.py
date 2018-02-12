#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import json
# import numpy as np

from os.path import expanduser
import matplotlib.pyplot as plt

from monster.misc import get_res_filepath
from monster.log import init_log

DATA_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def testplot(args):
    fn = args[0]
    with open(fn):
        d = json.load(open(fn, errors="ignore"))

    sd = sorted(
        [(k, v) for k, v in d.items()
         if v['bhattacharyya'] != 'nan' and v['cosine'] != '-nan'],
        key=lambda x: x[1]['bhattacharyya'])
    bhattacharyya = [float(i[1]["bhattacharyya"]) for i in sd]
    good_occ = [float(i[1]["good_occurrence"]) for i in sd]
    # bad_occ = [float(i[1]["bad_occurrence"]) for i in sd]

    # plot with various axes scales
    fig, axs = plt.subplots(1, 3)

    axs[0].set_yscale("log")
    axs[0].scatter(bhattacharyya, good_occ)
    axs[0].set_xlabel('bhattacharyya')
    axs[0].set_ylabel('good_occurrence')

    sd = sorted(
        [(k, v) for k, v in d.items()
         if v['cosine'] != 'nan' and v['cosine'] != '-nan'],
        key=lambda x: x[1]['cosine'])
    cosine = [float(i[1]["cosine"]) for i in sd]
    good_occ = [float(i[1]["good_occurrence"]) for i in sd]

    axs[1].set_yscale("log")
    axs[1].scatter(cosine, good_occ)
    axs[1].set_xlabel('cosine')
    axs[1].set_ylabel('good_occurrence')

    sd = sorted(
        [(k, v) for k, v in d.items()
         if v['bhattacharyya'] != 'nan' and v['cosine'] != 'nan'],
        key=lambda x: x[1]['bhattacharyya'])
    cosine = [float(i[1]["cosine"]) for i in sd]
    bhattacharyya = [float(i[1]["bhattacharyya"]) for i in sd]
    axs[2].scatter(cosine, bhattacharyya)
    axs[2].set_xlabel('cosine')
    axs[2].set_ylabel('bhattacharyya')

    plt.show()

    # parser = argparse.ArgumentParser(description="args for preprocess")
    # parser.add_argument("data_type", choices=["reddit", "hackforums", "darkode", "nulled"], type=str)
    # parser.add_argument(
    #     "-o", "--out_dir", help="output dir", type=str, default="")
    # parser.add_argument(
    #     "-d", "--debug", help="debug mode", action="store_true", default=False)
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     help="verbose mode",
    #     action="store_true",
    #     default=False)
    #
    # options = parser.parse_args(args)
    # log_config = dict(name=__file__, debug=options.debug)
    # out_dir = get_res_filepath(options.out_dir)
    # if options.verbose:
    #     log_config['console_verbosity'] = logging.INFO
    # logger = init_log(**log_config)
