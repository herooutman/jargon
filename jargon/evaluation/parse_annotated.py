#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import glob
import argparse
import os
import logging

from collections import defaultdict

from monster.misc import get_res_filepath
from ..utils.misc import tokenize
from monster.log import init_log

PREPROCESSED_DIR = os.path.abspath(
    os.path.join(get_res_filepath(), os.pardir, "preprocessing"))


def parse_annotated_impl(in_dir, out_file):
    res = defaultdict(set)
    for forum in os.listdir(in_dir):
        forum_path = os.path.join(in_dir, forum)
        if not os.path.isdir(forum_path):
            continue
        parse_forum(forum, forum_path, res)
    res = {k: list(v) for k, v in res.items()}
    with open(out_file, 'w') as fd:
        json.dump(res, fd, indent=2, sort_keys=True)
    logger.info("finished, output saved at '{}'".format(out_file))


def parse_forum(forum, forum_path, res):
    annotation_info = parse_annotations_folder(forum, forum_path)
    tokenized_path = os.path.join(forum_path, 'tokenized')
    code_pat = re.compile(r".+?(\d+)\.txt\.tok")
    if forum == 'nulled':
        annotation_type = None
        for fn in filter(lambda x: x.endswith("txt.tok"),
                         os.listdir(tokenized_path)):
            code = code_pat.match(fn).group(1)
            if code in annotation_info:
                with open(os.path.join(tokenized_path, fn)) as fd:
                    texts = fd.read()
                    for start, end, tagger in annotation_info[code]:
                        word = texts[start:end].strip().lower()
                        for token in tokenize(word):
                            if len(token) == 1:
                                continue
                            res[token].add((forum, str(annotation_type),
                                            tagger))

    else:
        for annotation_type in os.listdir(tokenized_path):
            type_dir = os.path.join(tokenized_path, annotation_type)
            if not os.path.isdir(type_dir):
                continue
            for fn in filter(
                    lambda x: x.endswith("txt.tok") and x.startswith('0-initiator'),
                    os.listdir(type_dir)):
                if forum == 'hackforums':
                    code = fn
                else:
                    code = code_pat.match(fn).group(1)

                if code in annotation_info:
                    with open(os.path.join(type_dir, fn)) as fd:
                        texts = fd.read()
                        for start, end, tagger in annotation_info[code]:
                            word = texts[start:end].strip().lower()
                            for token in tokenize(word):
                                if len(token) == 1:
                                    continue
                                res[token].add((forum, str(annotation_type),
                                                tagger))


def parse_annotations_folder(forum, forum_path):
    res = defaultdict(set)
    for annotations_fn in glob.iglob(
            os.path.join(forum_path, "annotations/*.txt")):
        with open(annotations_fn) as fd:
            for line in fd:
                line = line.strip()
                fields = line.split(" ")

                if forum == 'hackforums':
                    if len(fields) < 5:
                        continue
                    code, tagger, start, _, end = fields[0:5]
                else:
                    if len(fields) < 4:
                        continue
                    code, tagger, start, end = fields[0:4]
                res[code].add((int(start), int(end), tagger))
    return res


def parse_annotated(args):
    global logger
    parser = argparse.ArgumentParser(description="args for parse_annotated")
    parser.add_argument(
        "-i", "--in_dir", help="input dir", type=str, default="")
    parser.add_argument(
        "-o",
        "--out_file",
        help="output dir",
        type=str,
        default="annotations.json")
    parser.add_argument(
        "-w", "--workers", help="number of workers", type=int, default=10)
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
    out_file = get_res_filepath(fn=options.out_file)
    in_dir = os.path.join(PREPROCESSED_DIR, options.in_dir)
    if options.verbose:
        log_config['console_verbosity'] = logging.INFO
    logger = init_log(**log_config)

    parse_annotated_impl(in_dir=in_dir, out_file=out_file)
