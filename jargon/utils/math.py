import math
from numpy import dot
from gensim.matutils import unitvec


def bhattacharyya(v1, v2):
    bc = 0
    s1 = sum(v1)
    s2 = sum(v2)
    for i in range(len(v1)):
        bc += math.sqrt(v1[i] * v2[i] / s1 / s2)
    return -math.log(bc, 10)


def cosine(v1, v2):
    return 1 - float(
        dot(unitvec(v1), unitvec(v2)))
