#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "const.h"
#include "vocab.h"

typedef float real;

char syn0_file[MAX_STRING], syn1_file[MAX_STRING], corpus_file[MAX_STRING],
    vocab_file[MAX_STRING], output_file[MAX_STRING], prob_file[MAX_STRING],
    occur_file[MAX_STRING];
int debug_mode = 2, min_count = 5, min_reduce = 1, num_threads = 20, window = 5,
    binary = 0;

long long *occurrence;
long long layer1_size = -1, words_per_thread;
long long train_words = 0, word_count_actual = 0;
real exp_max, exp_min;
real *syn0, *syn1, *expTable;
real *probability;
clock_t start;

Vocab vocabulary;

// ========================================

void LoadVectors() {
  FILE *f;
  long long a, b;
  char ch;
  long long index, words, words2, layer1_size2;
  char word[MAX_STRING];

  printf("Loading bad vector (syn0) from '%s'\n", syn0_file);
  // syn0
  f = fopen(syn0_file, "rb");
  if (f == NULL) {
    printf("syn0 file not found\n");
    return; // shouldn't happen
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &layer1_size);
  printf("%lld vocab size, %lld words, %lld layer1_size\n",
         vocabulary.vocab_size, words, layer1_size);

  a = posix_memalign((void **)&syn0, 128,
                     (long long)vocabulary.vocab_size * layer1_size *
                         sizeof(real));
  if (syn0 == NULL) {
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

    index = SearchVocab(vocabulary, word);
    for (a = 0; a < layer1_size; a++) {
      if (binary) {
        fread(&syn0[a + index * layer1_size], sizeof(float), 1, f);
      } else {
        fscanf(f, "%f ", &syn0[a + index * layer1_size]);
      }
    }
  }
  fclose(f);
  printf("Loading good vector* (syn1) from '%s'\n", syn1_file);
  // syn1
  // ...
  // ...
  f = fopen(syn1_file, "rb");
  if (f == NULL) {
    printf("syn1 file not found\n");
    return; // shouldn't happen
  }
  fscanf(f, "%lld", &words2);
  fscanf(f, "%lld", &layer1_size2);
  if (words2 != words || layer1_size != layer1_size2) {
    printf("inconsistent good and bad files");
    return;
  }
  printf("%lld vocab size, %lld words, %lld size\n", vocabulary.vocab_size,
         words, layer1_size);

  a = posix_memalign((void **)&syn1, 128,
                     (long long)vocabulary.vocab_size * layer1_size *
                         sizeof(real));
  if (syn1 == NULL) {
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

    index = SearchVocab(vocabulary, word);
    // printf("got word %d:%s\n", index, word);
    for (a = 0; a < layer1_size; a++) {
      if (binary) {
        fread(&syn1[a + index * layer1_size], sizeof(float), 1, f);
      } else {
        fscanf(f, "%f ", &syn1[a + index * layer1_size]);
      }
    }
  }
  fclose(f);
}

void *ComputeProbabilityThread(void *id) {
  long long a, b, c;
  clock_t now;

  long long start_idx = (long long)id * words_per_thread;
  long long end_idx = start_idx + words_per_thread;
  if (end_idx > vocabulary.vocab_size)
    end_idx = vocabulary.vocab_size;

  for (a = start_idx; a < end_idx; a++) {
    for (b = 0; b < vocabulary.vocab_size; b++) {
      // compute probability[a][b], where a=center_word, b=context_word
      for (c = 0; c < layer1_size; c++) {
        probability[a * vocabulary.vocab_size + b] +=
            syn0[c + a * layer1_size] * syn1[c + b * layer1_size];
      }
      // now we have (v'T.v)
      if (probability[a * vocabulary.vocab_size + b] > MAX_EXP) {
        probability[a * vocabulary.vocab_size + b] = exp_max;
      } else if (probability[a * vocabulary.vocab_size + b] < -MAX_EXP) {
        probability[a * vocabulary.vocab_size + b] = exp_min;
      } else {
        // probability[a * vocabulary.vocab_size + b] = expTable[(
        //     int)((probability[a * vocabulary.vocab_size + b] + MAX_EXP) *
        //          (EXP_TABLE_SIZE / MAX_EXP / 2))];
        probability[a * vocabulary.vocab_size + b] =
            exp(probability[a * vocabulary.vocab_size + b]);
      }
      // now we have sigmoid(v'T.v) or exp(v'T.v)
    }
    //
    real sum = 0;
    for (b = 0; b < vocabulary.vocab_size; b++) {
      sum += probability[a * vocabulary.vocab_size + b];
    }

    if (sum > 0) {
      for (b = 0; b < vocabulary.vocab_size; b++) {
        probability[a * vocabulary.vocab_size + b] /= sum;
      }
    }
    // now we have p(b|a)
    if ((a - start_idx) % 5 == 4) {
      word_count_actual += 5;
      now = clock();
      printf("%cProcessed_words: %lld Progress: %.2f%%  "
             "Words/thread/sec: %.2f",
             13, word_count_actual,
             word_count_actual / (real)(vocabulary.vocab_size) * 100,
             word_count_actual /
                 ((real)(now - start + 1) / (real)CLOCKS_PER_SEC));
      fflush(stdout);
    }
  }
  pthread_exit(NULL);
}

void ComputeProbabilityTable() {
  long long a;
  LoadVectors();

  // probability
  // probability[w_center][w_context] = p(w_context|w_center)
  printf("Pre-computing probability...\n");
  printf("Trying to allocate memory: %.02f GB\n",
         vocabulary.vocab_size * vocabulary.vocab_size * sizeof(real) /
             (real)1048576 / 1024);
  // a = posix_memalign((void **)&probability, 128,
  //                    (long long)vocab_size * vocab_size * sizeof(real));
  probability = (real *)calloc(vocabulary.vocab_size * vocabulary.vocab_size,
                               sizeof(real));
  if (probability == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  // for (a = 0; a < vocab_size * vocab_size; a++) {
  //   probability[a] = 0;
  // }

  start = clock();
  words_per_thread = (long long)ceil(vocabulary.vocab_size / (real)num_threads);
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, ComputeProbabilityThread, (void *)a);
  for (a = 0; a < num_threads; a++)
    pthread_join(pt[a], NULL);
  printf("\nFinish computing probability table\n");
}

void ComputeOccurrenceTable() {
  long long a, c, word, center_word, context_word,
      sentence_length = 0, sentence_position = 0, sentence_index = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  FILE *fi = fopen(corpus_file, "rb");
  clock_t now;

  printf("Computing Occurrence in corpus file %s\n", corpus_file);
  FILE *fin = fopen(corpus_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  // occurrence
  printf("Trying to allocate memory for occurrence table: %.02f GB\n",
         vocabulary.vocab_size * vocabulary.vocab_size * sizeof(real) /
             (real)1048576 / 1024);
  // a = posix_memalign((void **)&occurrence, 128,
  //                    (long long)vocab_size * vocab_size * sizeof(long long));
  occurrence = (long long *)calloc(
      vocabulary.vocab_size * vocabulary.vocab_size, sizeof(long long));
  if (occurrence == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  // for (a = 0; a < vocab_size * vocab_size; a++) {
  //   occurrence[a] = 0;
  // }

  // start computing
  start = clock();
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cProgress: %.2f%%  Words/sec: %.2fk  ", 13,
               word_count_actual / (real)(train_words + 1) * 100,
               word_count_actual /
                   ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
    }

    // load a sentence if no sentence
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(vocabulary, fi);
        // printf("center word %lld, %s\n", center_word,
        // vocab[center_word].word);
        if (feof(fi))
          break;
        if (word == -1)
          continue;
        word_count++;
        if (word == 0)
          break;
        // TODO not sure if we need do subsampling
        // if (sample > 0) {
        //   real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
        //              (sample * train_words) / vocab[word].cn;
        //   next_random = next_random * (unsigned long long)25214903917 + 11;
        //   if (ran < (next_random & 0xFFFF) / (real)65536)
        //     continue;
        // }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH)
          break;
      }
      sentence_position = 0;
      sentence_index += 1;
    }

    // if at the end of the batch, new iteration
    if (feof(fi)) {
      break;
    }

    // in the middle of a sentence
    center_word = sen[sentence_position];
    if (center_word == -1)
      continue;
    // TODO init something for this sentence
    // skip gram prediction
    for (a = 0; a < window * 2 + 1; a++) {
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0)
          continue;
        if (c >= sentence_length)
          continue;
        context_word = sen[c];
        occurrence[center_word * vocabulary.vocab_size + context_word] += 1;
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  printf("\nFinish computing occurrence table\n");
}

void Predict() {
  FILE *fo;
  long long a, b;
  if (access(corpus_file, F_OK) == -1) {
    printf("Error: corpus file '%s' does not exist\n", corpus_file);
    return;
  }
  if (access(syn0_file, F_OK) == -1) {
    printf("Error: bad vector* (syn0) file '%s' does not exist\n", syn0_file);
    return;
  }
  if (access(syn1_file, F_OK) == -1) {
    printf("Error: good vector (syn1) file '%s' does not exist\n", syn1_file);
    return;
  }
  if (access(syn1_file, F_OK) == -1) {
    printf("Error: vocab file '%s' does not exist\n", vocab_file);
    return;
  }

  ReadVocab(vocabulary, vocab_file);
  ComputeProbabilityTable();
  if (prob_file[0] != 0) {
    fo = fopen(prob_file, "wb");
    fprintf(fo, "%lld\n", vocabulary.vocab_size);
    for (a = 0; a < vocabulary.vocab_size; a++) {
      fprintf(fo, "%s ", vocabulary.words[a].word);
      for (b = 0; b < vocabulary.vocab_size; b++) {
        fwrite(&probability[a * vocabulary.vocab_size + b], sizeof(real), 1,
               fo);
        // printf("%lf(%lld)\n", probability[a * vocabulary.vocab_size + b], a);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
    printf("probability matrix saved at '%s'\n", prob_file);
  }
  ComputeOccurrenceTable();
  if (occur_file[0] != 0) {
    fo = fopen(occur_file, "wb");
    fprintf(fo, "%lld\n", vocabulary.vocab_size);
    for (a = 0; a < vocabulary.vocab_size; a++) {
      fprintf(fo, "%s ", vocabulary.words[a].word);
      for (b = 0; b < vocabulary.vocab_size; b++) {
        fwrite(&occurrence[a * vocabulary.vocab_size + b], sizeof(long long), 1,
               fo);
        // printf("%lld, %lld: %lld\n", a, b, occurrence[a * vocab_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
    printf("occurrence matrix saved at '%s'\n", occur_file);
  }
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

// we are going to using trained vectors from bad corpus to predict in good
// corpus, so we use syn0 from bad and syn1 from good
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2VECTOR: prediction v 0.1c\n\n");
    printf("Options:\n");
    printf("\t-corpus <file>\n");
    printf("\t\tgood corpus file\n");
    printf("\t-good <file>\n");
    printf("\t\tvector* (syn1) file trained from good corpus\n");
    printf("\t-bad <file>\n");
    printf("\t\tvector (syn0) file trained from bad corpus\n");
    printf("\t-vocab <file>\n");
    printf("\t\tvocab file\n");
    printf("\t-output <file>\n");
    printf("\t\t<file> to store prediction results\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 20)\n");
    printf("\t-binary <int>\n");
    printf("\t\tLoad vectors with binary mode; default is 0 (off)\n");
    return 0;
  }
  output_file[0] = 0;
  corpus_file[0] = 0;
  prob_file[0] = 0;
  occur_file[0] = 0;
  syn0_file[0] = 0;
  syn1_file[0] = 0;
  vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-corpus", argc, argv)) > 0) {
    strcpy(corpus_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-good", argc, argv)) > 0) {
    strcpy(syn1_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-bad", argc, argv)) > 0) {
    strcpy(syn0_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-vocab", argc, argv)) > 0) {
    strcpy(vocab_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
    strcpy(prob_file, argv[i + 1]);
    strcat(prob_file, ".prob");
    strcpy(occur_file, argv[i + 1]);
    strcat(occur_file, ".occur");
  }
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
    window = atoi(argv[i + 1]);

  if (num_threads <= 0) {
    printf("number of threads must be larger than 0");
    return 1;
  }
  printf("window size: %d\n", window);
  // precompute expTable
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP); // Precompute the exp() table
    expTable[i] =
        expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  exp_max = 1;
  exp_min = 0;
  Predict();
  return 0;
}
