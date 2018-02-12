#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys


def eval_stats(args):
    from jargon.evaluation.stats import stats
    stats(args)


def parse_annotated(args):
    from jargon.evaluation.parse_annotated import parse_annotated
    parse_annotated(args)


def simulate(args):
    from jargon.preprocessing.simulate import simulate
    simulate(args)


def vocab(args):
    from jargon.word2vec.buildvocab import build_vocab
    build_vocab(args)


def train_text8(args):
    from jargon.word2vec.train_text8 import train
    train(args)


def train(args):
    from jargon.word2vec.train import train
    train(args)


def compare(args):
    from jargon.word2vec.compare import compare
    compare(args)


def compare_pred(args):
    from jargon.word2vec.compare_pred import compare_pred
    compare_pred(args)


def predict(args):
    from jargon.word2vec.predict import predict
    predict(args)


def preprocess(args):
    from jargon.preprocessing.preprocessor import preprocess
    preprocess(args)

def gettime(args):
    from jargon.gettime.preprocessor import preprocess
    preprocess(args)


def testplot(args):
    from jargon.visualize.testplot import testplot
    testplot(args)


def coverage(args):
    from jargon.stats.coverage import coverage
    coverage(args)


def text_stats(args):
    from jargon.stats.text_stats import get_text_stats
    get_text_stats(args)


def prepare(args):
    from jargon.preprocessing.prepare_training import prepare
    prepare(args)


def prepare_text2data(args):
    from jargon.preprocessing.prepare_text2data import prepare
    prepare(args)



def main():
    current_module = sys.modules[__name__]
    func_list = [
        x for x in dir(current_module)
        if callable(current_module.__dict__.get(x)) and x != "main"
    ]

    def print_func_list():
        sys.stderr.write(
            "avaliable functions: [{}]\n".format(", ".join(func_list)))

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python {} func_name.\n".format(sys.argv[0]))
        print_func_list()
        sys.exit(1)

    func_name = sys.argv[1]
    if func_name == "main":
        sys.stderr.write("Error: func_name cannot be main\n")
        sys.stderr.write("Usage: python {} func_name.\n".format(sys.argv[0]))
        print_func_list()
        sys.exit(1)
    try:
        module = getattr(current_module, func_name)
    except AttributeError:
        sys.stderr.write("Unknown module: {}\n".format(func_name))
        print_func_list()
        sys.exit(1)

    module(sys.argv[2:])


if __name__ == '__main__':
    main()
