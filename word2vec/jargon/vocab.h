#ifndef VOCAB_H
#define VOCAB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "const.h"
#define DEFAULT_MAX_SIZE 1000
#define DEFAULT_MIN_REDUCE 1

const int vocab_hash_size =
    30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
  long long cn, gcn, bcn; // count
  int *point;
  char *word, *code, codelen;
};

class Vocab {
public:
  long long vocab_size;
  long long vocab_max_size;
  long long train_words;
  long long good_train_words;
  long long bad_train_words;
  int min_count;
  int min_reduce;
  int *vocab_hash;
  struct vocab_word *words;

  Vocab() {
    vocab_max_size = DEFAULT_MAX_SIZE;
    min_reduce = DEFAULT_MIN_REDUCE;
    vocab_size = 0;
    train_words = 0;
    good_train_words = 0;
    bad_train_words = 0;
    words =
        (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  };
  void set_min_count(int minc) { min_count = minc; }
};

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++)
    hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found,
// returns -1
int SearchVocab(Vocab &vocab, char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab.vocab_hash[hash] == -1)
      return -1;
    if (!strcmp(word, vocab.words[vocab.vocab_hash[hash]].word))
      return vocab.vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(Vocab &vocab, char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING)
    length = MAX_STRING;
  vocab.words[vocab.vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab.words[vocab.vocab_size].word, word);
  vocab.words[vocab.vocab_size].cn = 0;
  vocab.vocab_size++;
  // Reallocate memory if needed
  if (vocab.vocab_size + 2 >= vocab.vocab_max_size) {
    vocab.vocab_max_size += 1000;
    vocab.words = (struct vocab_word *)realloc(
        vocab.words, vocab.vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab.vocab_hash[hash] != -1)
    hash = (hash + 1) % vocab_hash_size;
  vocab.vocab_hash[hash] = vocab.vocab_size - 1;
  return vocab.vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Reads a single word from a file, assuming space + tab + EOL to be word
// boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13)
      continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n')
          ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else
        continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1)
      a--; // Truncate too long words
  }
  word[a] = 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(Vocab &vocab) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab.words[1], vocab.vocab_size - 1, sizeof(struct vocab_word),
        VocabCompare);
  for (a = 0; a < vocab_hash_size; a++)
    vocab.vocab_hash[a] = -1;
  size = vocab.vocab_size;
  vocab.train_words = 0;
  vocab.good_train_words = 0;
  vocab.bad_train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab.words[a].cn < vocab.min_count) && (a != 0)) {
      vocab.vocab_size--;
      free(vocab.words[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab.words[a].word);
      while (vocab.vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
      vocab.vocab_hash[hash] = a;
      vocab.train_words += vocab.words[a].cn;
      vocab.good_train_words += vocab.words[a].gcn;
      vocab.bad_train_words += vocab.words[a].bcn;
    }
  }
  vocab.words = (struct vocab_word *)realloc(
      vocab.words, (vocab.vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab.vocab_size; a++) {
    vocab.words[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab.words[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(Vocab &vocab) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab.vocab_size; a++)
    if (vocab.words[a].cn > vocab.min_reduce) {
      vocab.words[b].cn = vocab.words[a].cn;
      vocab.words[b].word = vocab.words[a].word;
      b++;
    } else
      free(vocab.words[a].word);
  vocab.vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++)
    vocab.vocab_hash[a] = -1;
  for (a = 0; a < vocab.vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab.words[a].word);
    while (vocab.vocab_hash[hash] != -1)
      hash = (hash + 1) % vocab_hash_size;
    vocab.vocab_hash[hash] = a;
  }
  fflush(stdout);
  vocab.min_reduce++;
}

long long LearnVocabFromTrainFile(Vocab &vocab, char *train_file) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  long long file_size;
  for (a = 0; a < vocab_hash_size; a++)
    vocab.vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab.vocab_size = 0;
  AddWordToVocab(vocab, (char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    vocab.train_words++;
    if (vocab.train_words % 100000 == 0) {
      printf("%lldK%c", vocab.train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(vocab, word);
    if (i == -1) {
      a = AddWordToVocab(vocab, word);
      vocab.words[a].cn = 1;
    } else
      vocab.words[i].cn++;
    if (vocab.vocab_size > vocab_hash_size * 0.7)
      ReduceVocab(vocab);
  }
  SortVocab(vocab);
  printf("Vocab size: %lld\n", vocab.vocab_size);
  printf("Words in train file: %lld\n", vocab.train_words);

  file_size = ftell(fin);
  fclose(fin);
  return file_size;
}

void SaveVocab(Vocab &vocab, char *save_vocab_file) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab.vocab_size; i++)
    fprintf(fo, "%s %lld %lld %lld\n", vocab.words[i].word, vocab.words[i].cn,
            vocab.words[i].gcn, vocab.words[i].bcn);
  fclose(fo);
}

void ReadVocab(Vocab &vocab, char *read_vocab_file) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++)
    vocab.vocab_hash[a] = -1;
  vocab.vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin))
      break;
    a = AddWordToVocab(vocab, word);
    fscanf(fin, "%lld%c ", &vocab.words[a].cn, &c);
    fscanf(fin, "%lld%c ", &vocab.words[a].gcn, &c);
    fscanf(fin, "%lld%c", &vocab.words[a].bcn, &c);
    i++;
  }
  SortVocab(vocab);

  printf("Vocab size: %lld\n", vocab.vocab_size);
  printf("Words in train file: %lld\n", vocab.train_words);
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(Vocab &vocab, FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin))
    return -1;
  return SearchVocab(vocab, word);
}

#endif /* VOCAB_H */
