#ifndef UTIL_H
#define UTIL_H

#include "const.h"
#include "vocab.h"

long long read_vectors(char *fn, Vocab &vocab, float **res, int binary) {
  FILE *f;
  long long a, b, words, vector_size, index;
  char ch;
  char word[MAX_STRING];

  printf("Loading word vectors from '%s'\n", fn);
  // syn0
  f = fopen(fn, "rb");
  if (f == NULL) {
    printf("Error: vector file not found\n");
    exit(1);
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &vector_size);
  printf("vocab size: %lld, words: %lld, vector_size: %lld\n", vocab.vocab_size,
         words, vector_size);
  a = posix_memalign((void **)res, 128,
                     (long long)vocab.vocab_size * vector_size * sizeof(float));

  if (*res == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      ch = fgetc(f);
      if (feof(f) || (ch == ' ')) {
        break;
      }
      if ((a < MAX_STRING) && (ch != '\n')) {
        word[a] = ch;
        a++;
      }
    }
    word[a] = 0;

    index = SearchVocab(vocab, word);
    for (a = 0; a < vector_size; a++) {
      if (binary) {
        fread(&((*res)[a + index * vector_size]), sizeof(float), 1, f);
      } else {
        fscanf(f, "%f ", &((*res)[a + index * vector_size]));
      }
    }
  }
  fclose(f);
  return vector_size;
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

#endif /* UTIL_H */
