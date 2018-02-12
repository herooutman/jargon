import time
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import logging

from collections import Counter

TIME_FORMAT_PATTERN = "%Y%m%d%H%M%S"


def simulate(args):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="args for prediction")
    parser.add_argument(
        "-p",
        "--percent",
        help="percentage of the word occurrences to be change (0-1)",
        type=float,
        default=1.0)
    parser.add_argument(
        "-c",
        "--count",
        help="number of words to be changed (0-100)",
        type=int,
        default=5)
    parser.add_argument(
        "-o",
        "--outtag",
        help="tag of output file",
        type=str,
        default="default")
    parser.add_argument("-i", "--input", help="input corpus", type=str)

    options = parser.parse_args(args)

    if options.count < 0 or options.count > 100:
        logging.error("Error: invalid count {}.".format(options.count))
        return 1
    if options.percent < 0 or options.percent > 1:
        logging.error("Error: invalid percentage {}.".format(options.percent))
        return 1
    if options.input and os.path.isfile(options.input):
        pass
    else:
        logging.error("Error: prob file '{}' not found".format(options.input))
        return 1

    outfile = "{}.simulated.{}".format(options.input, options.outtag)
    ansfile = "{}.answer".format(outfile)

    with open(options.input, "r") as fd:
        lines = fd.readlines()
    ctr = Counter()
    for line in lines:
        words = line.split()
        ctr.update(words)
    logging.info("Corpus file '{}' loaded".format(options.input))

    selections = random.sample([x for x in ctr if ctr[x] > 250], options.count)

    simu_info = dict()
    for x in selections:
        while True:
            target = random.choice(list(ctr.keys()))
            if ctr[x] / 10 <= ctr[target] <= ctr[x] * 10 and target not in selections:
                break

        simu_info[x] = (x, ctr[x], target, ctr[target])

    with open(outfile, "w") as fd:
        for line in lines:
            newline = []
            for word in words:
                if word in simu_info:
                    test = random.random()
                    if test < options.percent:
                        newline.append(simu_info[word][2])
                    else:
                        newline.append(word)
                else:
                    newline.append(word)
            fd.write(" ".join(newline))
            fd.write("\n")

    with open(ansfile, "w") as fd:
        for i in simu_info.values():
            fd.write(str(i))
            fd.write("\n")
    logging.info(
        "finished, results saved at '{}' and '{}'.".format(outfile, ansfile))
