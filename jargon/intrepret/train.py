#!/bin/python
# Author: Jiang Guo

import cPickle
import numpy as np
from numpy import transpose

import time

# rng can be moved to main func
rng = np.random.RandomState(1234)

CONSTANT = 1
OPTIMAL = 2
INVSCALING = 3
PA1 = 4
PA2 = 5


class Trainer(object):
    ''' This trainer optimize the following objective function:
            0.5 * || W*X - Y ||^2
        Training is performed using stochastic gradient descent.
    '''

    def __init__(self, dim=300):

        self.dim = dim  # dimension of embedding
        self.eta_max = 0.001  # max learning rate
        self.eta_min = 0.00001  # min learning rate
        self.max_epochs = 900

        #self.weights = np.asmatrix(rng.uniform(low = -0.01, high = 0.01,
        #    size = (self.dim, self.dim)), dtype = float)
        self.weights = np.asmatrix(np.zeros((self.dim, self.dim)))
        # self.bias = np.asmatrix(np.zeros((self.dim, 1)))

    def load_model(self, model_path):
        self.weights = cPickle.load(open(model_path, "rb"))

    def set_training_data(self, X, Y):
        self.train_x = np.asmatrix(X)
        self.train_y = np.asmatrix(Y)

    def set_heldout_data(self, X, Y):
        self.heldout_x = np.asmatrix(X)
        self.heldout_y = np.asmatrix(Y)

    def compute_training_loss(self):
        assert self.train_x is not None
        assert self.train_y is not None

        train_p = self.weights * self.train_x.transpose()
        loss = 0.5 * (np.linalg.norm(
            (train_p - self.train_y.transpose()), 'fro')**2)

        return loss

    def compute_heldout_loss(self):
        assert self.heldout_x is not None
        assert self.heldout_y is not None

        heldout_p = self.weights * self.heldout_x.transpose()
        loss = 0.5 * (np.linalg.norm(
            (heldout_p - self.heldout_y.transpose()), 'fro')**2)

        return loss

    def compute_heldout_inv_loss(self):
        assert self.heldout_x is not None
        assert self.heldout_y is not None

        heldout_p = self.weights * self.heldout_y.transpose()
        loss = 0.5 * (np.linalg.norm(
            (heldout_p - self.heldout_x.transpose()), 'fro')**2)

        return loss

    def train(self, learning_rate, logfile):
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[1] == self.train_y.shape[1]
        fp_log = open(logfile, "w")

        N = self.train_x.shape[0]

        self.step = (self.eta_max - self.eta_min) / (self.max_epochs * N)
        self.eta = self.eta_max

        training_loss = 0
        t_start = time.time()

        heldout_loss = 10000
        print(
            "Epoch    train loss     heldout loss    heldout inv loss    learning rate    Time"
        )
        print(
            "================================================================================="
        )

        for epoch in range(self.max_epochs):

            t_end = time.time()
            if epoch % 100 == 0:
                training_loss = self.compute_training_loss()
            else:
                training_loss = "-"
            pre_heldout_loss = heldout_loss
            heldout_loss = self.compute_heldout_loss()
            heldout_inv_loss = self.compute_heldout_inv_loss()

            if heldout_loss >= pre_heldout_loss:
                break

            # print >> sys.stderr, "Epoch %d: train loss = %f, heldout loss = %f, learning rate = %6f (%.2f s)" % \
            #         (epoch+1, training_loss, heldout_loss, self.eta, t_end-t_start)
            print("%5d\t%10s\t%12f\t%12f\t%6f\t%.2f" %
                  (epoch + 1, str(training_loss), heldout_loss,
                   heldout_inv_loss, self.eta, t_end - t_start))

            for ii in xrange(N):

                if learning_rate == OPTIMAL:
                    # self.eta = 1.0 / (alpha * t)
                    # self.eta = eta0 * pow(alpha, float(t) / N)
                    self.eta = self.eta - self.step

                x = self.train_x[ii, :].transpose()
                gold_y = self.train_y[ii, :].transpose()

                pred_y = self.weights * x
                delta_weights = (pred_y - gold_y) * x.transpose()
                self.weights -= self.eta * delta_weights

                # if learning_rate == OPTIMAL:
                #     t += 1

    def save_model(self, path):
        cPickle.dump(self.weights, open(path, "wb"), protocol=2)


import sys
import os.path
import glob

from multiprocessing import Pool, cpu_count


def process(obj_dir, dim):
    print "[start] training [%s] | dim=%d" % (os.path.basename(obj_dir), dim)

    f_train_x = os.path.join(obj_dir, "train.x")
    f_train_y = os.path.join(obj_dir, "train.y")
    f_heldout_x = os.path.join(obj_dir, "heldout.x")
    f_heldout_y = os.path.join(obj_dir, "heldout.y")

    train_x, keys = cPickle.load(open(f_train_x))
    train_y, keys = cPickle.load(open(f_train_y))
    heldout_x, keys = cPickle.load(open(f_heldout_x))
    heldout_y, keys = cPickle.load(open(f_heldout_y))

    trainer = Trainer(dim)
    trainer.set_training_data(train_x, train_y)
    trainer.set_heldout_data(heldout_x, heldout_y)

    log_path = os.path.join(obj_dir, "train.log")
    model_path = os.path.join(obj_dir, "model")
    if os.path.isfile(model_path):
        trainer.load_model(model_path)

    trainer.train(learning_rate=OPTIMAL, logfile=log_path)
    trainer.save_model(model_path)

    print "[complete] training [%s]" % (os.path.basename(obj_dir))


if __name__ == '__main__':

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print >> sys.stderr,\
                 "[Usage] %s data_root dim [thread_num]" % (sys.argv[0])
        sys.exit()

    if len(sys.argv) == 4:
        n_thread = int(sys.argv[3])
    else:
        n_thread = cpu_count()

    dim = int(sys.argv[2])

    proc_pool = Pool(processes=n_thread)
    dfolders = glob.glob(os.path.join(sys.argv[1], "*"))
    for folder in dfolders:
        if folder.endswith("kmeans.mm"): continue
        result = proc_pool.apply_async(process, (folder, dim))
    proc_pool.close()
    proc_pool.join()

    if result.successful():
        print 'All models trained completely.'
