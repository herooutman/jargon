#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "const.h"
#include "vocab.h"

#define MAX_STRING 100
#define MAX_CODE_LENGTH 40
#define OUTPUT_DIR "/u/kanyuan/sbout/jargon/word2vec/jargon/"

char good_file[MAX_STRING], bad_file[MAX_STRING], output_file[MAX_STRING];
long long file_size = 0, train_words = 0;
int debug_mode = 2, min_count = 5, min_reduce = 1, common_only = 0;
Vocab vocabulary, vocabulary_common;

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

void GetCommonVocab() {
  long long a, i, j;
  for (a = 0; a < vocab_hash_size; a++)
    vocabulary_common.vocab_hash[a] = -1;
  vocabulary_common.vocab_size = 0;
  AddWordToVocab(vocabulary_common, (char *)"</s>");
  for (a = 0; a < vocabulary.vocab_size; a++) {
    struct vocab_word word = vocabulary.words[a];
    i = SearchVocab(vocabulary_common, word.word);
    if (i == -1) {
      if (word.gcn > 0 && word.bcn > 0) {
        j = AddWordToVocab(vocabulary_common, word.word);
        vocabulary_common.words[j].cn = word.cn;
        vocabulary_common.words[j].gcn = word.gcn;
        vocabulary_common.words[j].bcn = word.bcn;
      }
    } else {
      vocabulary_common.words[i].cn = word.cn;
      vocabulary_common.words[i].gcn = word.gcn;
      vocabulary_common.words[i].bcn = word.bcn;
    }
  }
  SortVocab(vocabulary_common);
  if (debug_mode > 0) {
    printf("Common vocab size: %lld\n", vocabulary_common.vocab_size);
  }
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *goodin, *badin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++)
    vocabulary.vocab_hash[a] = -1;
  vocabulary.vocab_size = 0;
  AddWordToVocab(vocabulary, (char *)"</s>");
  goodin = fopen(good_file, "rb");
  if (goodin != NULL) {
    while (1) {
      ReadWord(word, goodin);
      if (feof(goodin))
        break;
      train_words++;
      if ((debug_mode > 1) && (train_words % 100000 == 0)) {
        printf("%lldK%c", train_words / 1000, 13);
        fflush(stdout);
      }
      i = SearchVocab(vocabulary, word);
      if (i == -1) {
        a = AddWordToVocab(vocabulary, word);
        vocabulary.words[a].cn = 1;
        vocabulary.words[a].gcn = 1;
        vocabulary.words[a].bcn = 0;
      } else {
        vocabulary.words[i].cn++;
        vocabulary.words[i].gcn++;
      }
      if (vocabulary.vocab_size > vocab_hash_size * 0.7)
        ReduceVocab(vocabulary);
    }
    fclose(goodin);
  }
  badin = fopen(bad_file, "rb");
  if (badin != NULL) {
    while (1) {
      ReadWord(word, badin);
      if (feof(badin))
        break;
      train_words++;
      if ((debug_mode > 1) && (train_words % 100000 == 0)) {
        printf("%lldK%c", train_words / 1000, 13);
        fflush(stdout);
      }
      i = SearchVocab(vocabulary, word);
      if (i == -1) {
        a = AddWordToVocab(vocabulary, word);
        vocabulary.words[a].cn = 1;
        vocabulary.words[a].gcn = 0;
        vocabulary.words[a].bcn = 1;
      } else {
        vocabulary.words[i].cn++;
        vocabulary.words[i].bcn++;
      }
      if (vocabulary.vocab_size > vocab_hash_size * 0.7)
        ReduceVocab(vocabulary);
    }
    fclose(badin);
  }
  SortVocab(vocabulary);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocabulary.vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR UTIL: combined vocab v 0.1c\n\n");
    printf("Options:\n");
    printf("\t-good <file>\n");
    printf("\t\tgood forum's text file\n");
    printf("\t-bad <file>\n");
    printf("\t\tbad forum's text file\n");
    printf("\t-output <file>\n");
    printf("\t\t<file> to store vocab\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; "
           "default is 5\n");
    printf("\t-common-only <int>\n");
    printf("\t\tThis will only keep words that appear in both corpus; default "
           "is 0\n");
    return 0;
  }
  output_file[0] = 0;
  if ((i = ArgPos((char *)"-good", argc, argv)) > 0)
    strcpy(good_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-bad", argc, argv)) > 0)
    strcpy(bad_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-common-only", argc, argv)) > 0)
    common_only = atoi(argv[i + 1]);
  vocabulary.set_min_count(min_count);
  vocabulary_common.set_min_count(min_count);
  LearnVocabFromTrainFile();
  if (common_only) {
    GetCommonVocab();
    SaveVocab(vocabulary_common, output_file);
  } else {
    SaveVocab(vocabulary, output_file);
  }
  return 0;
}
