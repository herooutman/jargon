#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <math.h>

float CosineDistance(float *v1, float *v2, int n) {
  int a;
  float ab = 0, a2 = 0, b2 = 0;
  for (a = 0; a < n; a++) {
    ab += v1[a] * v2[a];
    a2 += v1[a] * v1[a];
    b2 += v2[a] * v2[a];
  }
  return 1 - (ab / (sqrt(a2) * sqrt(b2)));
}

float HammingDistance(float *v1, float *v2, int n) {
  int a;
  float dist = 0;
  for (a = 0; a < n; a++) {
    dist += fabs(v1[a] - v2[a]);
  }
  return dist;
}
#endif /* MATHUTIL_H */
