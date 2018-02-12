//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "const.h"
#include "vocab.h"

typedef float real; // Precision of float numbers

char good_file[MAX_STRING], bad_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

int binary = 0, cbow = 1, debug_mode = 2, window = 5, num_threads = 12;

long long layer1_size = 100;
long long word_count_actual = 0, iter = 5, good_file_size = 0,
          bad_file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0_good, *syn0_bad, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;
Vocab vocabulary;

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocabulary.vocab_size; a++)
    train_words_pow += pow(vocabulary.words[a].cn, power);
  i = 0;
  d1 = pow(vocabulary.words[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocabulary.words[i].cn, power) / train_words_pow;
    }
    if (i >= vocabulary.vocab_size)
      i = vocabulary.vocab_size - 1;
  }
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count =
      (long long *)calloc(vocabulary.vocab_size * 2 + 1, sizeof(long long));
  long long *binary =
      (long long *)calloc(vocabulary.vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node =
      (long long *)calloc(vocabulary.vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocabulary.vocab_size; a++)
    count[a] = vocabulary.words[a].cn;
  for (a = vocabulary.vocab_size; a < vocabulary.vocab_size * 2; a++)
    count[a] = 1e15;
  pos1 = vocabulary.vocab_size - 1;
  pos2 = vocabulary.vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a
  // time
  for (a = 0; a < vocabulary.vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocabulary.vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocabulary.vocab_size + a;
    parent_node[min2i] = vocabulary.vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  // for each walk to leaf (word node)
  for (a = 0; a < vocabulary.vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      // point[i]: i-th parent
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocabulary.vocab_size * 2 - 2)
        break;
    }
    vocabulary.words[a].codelen = i;
    // root?
    vocabulary.words[a].point[0] = vocabulary.vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocabulary.words[a].code[i - b - 1] = code[b];
      // vocab[x].point[i]: i-th node from root in the walk
      vocabulary.words[a].point[i - b] = point[b] - vocabulary.vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

/*
  syn0, vocab_size * layer1_size, init randomly ( ),
  syn1, if using Hierarchical Softmax, same size as syn0, init with 0
  syn1neg, if using negative sampling, same size as syn0, init with 0
  CreateBinaryTree
*/
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0_good, 128,
                     (long long)vocabulary.vocab_size * layer1_size *
                         sizeof(real));
  if (syn0_good == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  a = posix_memalign((void **)&syn0_bad, 128,
                     (long long)vocabulary.vocab_size * layer1_size *
                         sizeof(real));
  if (syn0_bad == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  if (hs) {
    a = posix_memalign((void **)&syn1, 128,
                       (long long)vocabulary.vocab_size * layer1_size *
                           sizeof(real));
    if (syn1 == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocabulary.vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1[a * layer1_size + b] = 0;
  }
  if (negative > 0) {
    a = posix_memalign((void **)&syn1neg, 128,
                       (long long)vocabulary.vocab_size * layer1_size *
                           sizeof(real));
    if (syn1neg == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (a = 0; a < vocabulary.vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocabulary.vocab_size; a++)
    for (b = 0; b < layer1_size; b++) {
      // Linear congruential generator
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0_good[a * layer1_size + b] =
          (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      syn0_bad[a * layer1_size + b] = syn0_good[a * layer1_size + b];
    }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0,
                                          sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi_good = fopen(good_file, "rb");
  FILE *fi_bad = fopen(bad_file, "rb");

  bool next_iter = false;
  bool train_good;
  FILE *fi;
  long long file_size;
  long long train_words;
  real *syn0;

  if ((long long)id % 2 == 0) {
    train_good = true;
    fi = fi_good;
    file_size = good_file_size;
    train_words = vocabulary.good_train_words;
    syn0 = syn0_good;
  } else {
    train_good = false;
    fi = fi_bad;
    file_size = bad_file_size;
    train_words = vocabulary.bad_train_words;
    syn0 = syn0_bad;
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  printf("\nThread-%lld, fseek %s file, at %lld of %lld", (long long)id,
         (fi == fi_good ? "good" : "bad"),
         file_size / (long long)num_threads * (long long)id, file_size);
  while (1) {
    // print progress and update alpha
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
               alpha,
               word_count_actual / (real)(iter * vocabulary.train_words + 1) *
                   100,
               word_count_actual /
                   ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha =
          starting_alpha *
          (1 - word_count_actual / (real)(iter * vocabulary.train_words + 1));
      if (alpha < starting_alpha * 0.0001)
        alpha = starting_alpha * 0.0001;
    }
    // load a sentence if no sentence
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(vocabulary, fi);
        if (feof(fi))
          break;
        if (word == -1)
          continue;
        word_count++;
        if (word == 0)
          break;
        // The subsampling randomly discards frequent words while keeping the
        // ranking same
        if (sample > 0) {
          real ran = (sqrt(vocabulary.words[word].cn /
                           (sample * vocabulary.train_words)) +
                      1) *
                     (sample * vocabulary.train_words) /
                     vocabulary.words[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536)
            continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH)
          break;
      }
      sentence_position = 0;
    }

    // if at the end of the batch, new iteration
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      if (train_good == true) {
        train_good = false;
        fi = fi_bad;
        file_size = bad_file_size;
        train_words = vocabulary.bad_train_words;
        syn0 = syn0_bad;
      } else {
        train_good = true;
        fi = fi_good;
        file_size = good_file_size;
        train_words = vocabulary.good_train_words;
        syn0 = syn0_good;
      }
      if (next_iter == true) {
        next_iter = false;
        local_iter--; // next iteration
      } else {
        next_iter = true;
      }
      if (local_iter == 0)
        break;
      printf("\nThread-%lld, processed words %lld, sent len %lld, iter %lld, "
             "corpus %s, fseek "
             "%s file, at %lld of "
             "%lld\n",
             (long long)id, word_count, sentence_length,
             (iter - local_iter + 1), (train_good ? "good" : "bad"),
             (fi == fi_good ? "good" : "bad"),
             file_size / (long long)num_threads * (long long)id, file_size);
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    // in the middle of a sentence
    word = sen[sentence_position];
    if (word == -1)
      continue;
    for (c = 0; c < layer1_size; c++)
      neu1[c] = 0;
    for (c = 0; c < layer1_size; c++)
      neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    // randomly choose b from window
    b = next_random % window;

    if (cbow) { // train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0)
            continue;
          if (c >= sentence_length)
            continue;
          last_word = sen[c];
          if (last_word == -1)
            continue;
          for (c = 0; c < layer1_size; c++)
            neu1[c] += syn0[c + last_word * layer1_size];
          cw++;
        }
      if (cw) {
        for (c = 0; c < layer1_size; c++)
          neu1[c] /= cw;
        if (hs)
          for (d = 0; d < vocabulary.words[word].codelen; d++) {
            f = 0;
            l2 = vocabulary.words[word].point[d] * layer1_size;
            // Propagate hidden -> output
            for (c = 0; c < layer1_size; c++)
              f += neu1[c] * syn1[c + l2];
            if (f <= -MAX_EXP)
              continue;
            else if (f >= MAX_EXP)
              continue;
            else
              f = expTable[(int)((f + MAX_EXP) *
                                 (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocabulary.words[word].code[d] - f) * alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; c++)
              neu1e[c] += g * syn1[c + l2];
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; c++)
              syn1[c + l2] += g * neu1[c];
          }
        // NEGATIVE SAMPLING
        if (negative > 0)
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0)
                target = next_random % (vocabulary.vocab_size - 1) + 1;
              if (target == word)
                continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++)
              f += neu1[c] * syn1neg[c + l2];
            if (f > MAX_EXP)
              g = (label - 1) * alpha;
            else if (f < -MAX_EXP)
              g = (label - 0) * alpha;
            else
              g = (label - expTable[(int)((f + MAX_EXP) *
                                          (EXP_TABLE_SIZE / MAX_EXP / 2))]) *
                  alpha;
            for (c = 0; c < layer1_size; c++)
              neu1e[c] += g * syn1neg[c + l2];
            for (c = 0; c < layer1_size; c++)
              syn1neg[c + l2] += g * neu1[c];
          }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++)
          if (a != window) {
            c = sentence_position - window + a;
            if (c < 0)
              continue;
            if (c >= sentence_length)
              continue;
            last_word = sen[c];
            if (last_word == -1)
              continue;
            for (c = 0; c < layer1_size; c++)
              syn0[c + last_word * layer1_size] += neu1e[c];
          }
      }
    } else { // train skip-gram
      // randomly choose b from window
      // a : [b, window * 2 - b], a != window
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          // c : [pos - window + b, pos + window - b], c != pos
          c = sentence_position - window + a;
          if (c < 0)
            continue;
          if (c >= sentence_length)
            continue;
          last_word = sen[c];
          if (last_word == -1)
            continue;
          // offset of last word
          l1 = last_word * layer1_size;
          // neu1e reset for every word
          for (c = 0; c < layer1_size; c++)
            neu1e[c] = 0;

          // HIERARCHICAL SOFTMAX
          if (hs) {
            for (d = 0; d < vocabulary.words[word].codelen; d++) {
              f = 0;
              // offset of d-th node from root
              l2 = vocabulary.words[word].point[d] * layer1_size;
              // Propagate hidden -> output
              // f = syn0[last_word] . syn1[dth_node]
              for (c = 0; c < layer1_size; c++)
                f += syn0[c + l1] * syn1[c + l2];
              if (f <= -MAX_EXP)
                continue;
              else if (f >= MAX_EXP)
                continue;
              else
                // sigmoid function
                // f = exp(f) / (exp(f)+1)
                f = expTable[(int)((f + MAX_EXP) *
                                   (EXP_TABLE_SIZE / MAX_EXP / 2))];
              // 'g' is the gradient multiplied by the learning rate
              // g = 0 if one of them is 1
              // g = 1 if both 0
              // g = -1 if both 1
              // ???
              g = (1 - vocabulary.words[word].code[d] - f) * alpha;
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++)
                neu1e[c] += g * syn1[c + l2];
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++)
                syn1[c + l2] += g * syn0[c + l1];
            }
          }

          // NEGATIVE SAMPLING
          // Rather than performing backpropagation for every word in our
          // vocabulary, we only perform it for a few words (the number of words
          // is given by 'negative').
          // These words are selected using a "unigram" distribution, which is
          // generated in the function InitUnigramTable
          if (negative > 0) {
            for (d = 0; d < negative + 1; d++) {
              // On the first iteration, we're going to train the positive
              // sample.
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                // On the other iterations, we'll train the negative samples.
                next_random =
                    next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0)
                  target = next_random % (vocabulary.vocab_size - 1) + 1;
                if (target == word)
                  continue;
                label = 0;
              }
              // now we have a target word
              // l2: target word offset
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++)
                f += syn0[c + l1] * syn1neg[c + l2];
              // f is the input of the softmax layer (inner product of two
              // vectors)
              // prediction = exp(f) / (exp(f) + 1)
              if (f > MAX_EXP)
                g = (label - 1) * alpha;
              else if (f < -MAX_EXP)
                g = (label - 0) * alpha;
              else
                g = (label - expTable[(int)((f + MAX_EXP) *
                                            (EXP_TABLE_SIZE / MAX_EXP / 2))]) *
                    alpha;
              for (c = 0; c < layer1_size; c++)
                neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++)
                syn1neg[c + l2] += g * syn0[c + l1];
            }
          }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++)
            syn0[c + l1] += neu1e[c];
        }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/*
  load vocab
  InitNet
  if using negative sampling, InitUnigramTable
  start threads -> TrainModelThread
  output: a: 0 - vocab_size, b: 0 - layer1_size, syn0[a * layer1_size + b]
  if classes > 0, do k-means clustering
*/
void TrainModel() {
  long a, b;
  FILE *fo_good, *fo_bad, *fo1, *fo1neg, *fgood, *fbad;
  char output_file1[MAX_STRING];
  char output_file1neg[MAX_STRING];
  char good_output_file[MAX_STRING];
  char bad_output_file[MAX_STRING];
  good_output_file[0] = 0;
  bad_output_file[0] = 0;
  output_file1[0] = 0;
  output_file1neg[0] = 0;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using good file '%s' and bad file '%s'\n",
         good_file, bad_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) {
    printf("read vocab\n");
    ReadVocab(vocabulary, read_vocab_file);
    // get good file_size
    printf("getting training files size\n");
    fgood = fopen(good_file, "rb");
    if (fgood == NULL) {
      printf("ERROR: good data file not found!\n");
      exit(1);
    }
    fseek(fgood, 0, SEEK_END);
    good_file_size = ftell(fgood);
    fbad = fopen(bad_file, "rb");
    fclose(fgood);

    fbad = fopen(bad_file, "rb");
    if (fbad == NULL) {
      printf("ERROR: bad data file not found!\n");
      exit(1);
    }
    fseek(fbad, 0, SEEK_END);
    bad_file_size = ftell(fbad);
    fbad = fopen(bad_file, "rb");
    fclose(fbad);
  } else {
    printf("ERROR: vocab file not found!\n");
    exit(1);
  }
  if (save_vocab_file[0] != 0)
    SaveVocab(vocabulary, save_vocab_file);
  if (output_file[0] == 0)
    return;
  InitNet();
  if (negative > 0)
    InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++)
    pthread_join(pt[a], NULL);
  printf("\n");

  if (binary){
    strcat(output_file, ".binary");
  }
  strcpy(good_output_file, output_file);
  strcat(good_output_file, ".good.syn0");
  fo_good = fopen(good_output_file, "wb");
  strcpy(bad_output_file, output_file);
  strcat(bad_output_file, ".bad.syn0");
  fo_bad = fopen(bad_output_file, "wb");
  // Save the word vectors (syn0)
  fprintf(fo_good, "%lld %lld\n", vocabulary.vocab_size, layer1_size);
  fprintf(fo_bad, "%lld %lld\n", vocabulary.vocab_size, layer1_size);
  for (a = 0; a < vocabulary.vocab_size; a++) {
    fprintf(fo_good, "%s ", vocabulary.words[a].word);
    fprintf(fo_bad, "%s ", vocabulary.words[a].word);
    if (binary)
      for (b = 0; b < layer1_size; b++) {
        fwrite(&syn0_good[a * layer1_size + b], sizeof(real), 1, fo_good);
        fwrite(&syn0_bad[a * layer1_size + b], sizeof(real), 1, fo_bad);
      }
    else
      for (b = 0; b < layer1_size; b++) {
        fprintf(fo_good, "%lf ", syn0_good[a * layer1_size + b]);
        fprintf(fo_bad, "%lf ", syn0_bad[a * layer1_size + b]);
      }
    fprintf(fo_good, "\n");
    fprintf(fo_bad, "\n");
  }
  printf("Word vector saved at '%s' and '%s'\n", good_output_file,
         bad_output_file);
  // Save syn1
  if (hs) {
    strcpy(output_file1, output_file);
    strcat(output_file1, ".syn1");
    fo1 = fopen(output_file1, "wb");
    fprintf(fo1, "%lld %lld\n", vocabulary.vocab_size, layer1_size);
    for (a = 0; a < vocabulary.vocab_size; a++) {
      fprintf(fo1, "%s ", vocabulary.words[a].word);
      if (binary)
        for (b = 0; b < layer1_size; b++)
          fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo1);
      else
        for (b = 0; b < layer1_size; b++)
          fprintf(fo1, "%lf ", syn1[a * layer1_size + b]);
      fprintf(fo1, "\n");
    }
    fclose(fo1);
    printf("Hierarchical matrix saved at '%s'\n", output_file1);
  }
  // Save syn1neg
  if (negative > 0) {
    strcpy(output_file1neg, output_file);
    strcat(output_file1neg, ".syn1neg");
    fo1neg = fopen(output_file1neg, "wb");
    fprintf(fo1neg, "%lld %lld\n", vocabulary.vocab_size, layer1_size);
    for (a = 0; a < vocabulary.vocab_size; a++) {
      fprintf(fo1neg, "%s ", vocabulary.words[a].word);
      if (binary)
        for (b = 0; b < layer1_size; b++)
          fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo1neg);
      else
        for (b = 0; b < layer1_size; b++)
          fprintf(fo1neg, "%lf ", syn1neg[a * layer1_size + b]);
      fprintf(fo1neg, "\n");
    }
    fclose(fo1neg);
    printf("Negative matrix saved at '%s'\n", output_file1neg);
  }
  fclose(fo_good);
  fclose(fo_bad);
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

int main(int argc, char **argv) {
  int i;
  int min_count = DEFAULT_MIN_COUNT;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf(
        "\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with "
           "higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range "
           "is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 "
           "- 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; "
           "default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram "
           "and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number "
           "of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf(
        "\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf(
        "\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from "
           "the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for "
           "skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 "
           "-sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
    layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-good", argc, argv)) > 0)
    strcpy(good_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-bad", argc, argv)) > 0)
    strcpy(bad_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
    strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
    strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
    debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0)
    cbow = atoi(argv[i + 1]);
  if (cbow)
    alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
    alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
    window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
    sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0)
    hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
    negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
    iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
    min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0)
    classes = atoi(argv[i + 1]);

  vocabulary.set_min_count(min_count);
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) *
                      MAX_EXP); // Precompute the exp() table
    expTable[i] =
        expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
