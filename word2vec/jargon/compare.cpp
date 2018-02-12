#include <fstream>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <unordered_map>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
// #include <jansson.h>

#include "const.h"
#include "vocab.h"
// Short alias for this namespace
namespace pt = boost::property_tree;

typedef float real;

const long long N = 40;

char prob_file[MAX_STRING], occur_file[MAX_STRING], output_file[MAX_STRING],
    vocab_file[MAX_STRING];
int num_threads = 20;
std::unordered_map<std::string, real> distance;
pt::ptree root;
Vocab vocab;

void GetWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13)
      continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        break;
      } else {
        continue;
      }
    }
    word[a] = ch;
    a++;
  }
  word[a] = 0;
}

float BhattacharyyaDistance(real *probability, long long *occurrence, int n) {
  int a;
  real sum = 0, bc = 0;
  for (a = 0; a < n; a++) {
    sum += occurrence[a];
  }
  for (a = 0; a < n; a++) {
    bc += sqrtf(probability[a] * occurrence[a] / sum);
  }
  return -log(bc);
}

float CosineDistance(real *probability, long long *occurrence, int n) {
  int a;
  real ab = 0, a2 = 0, b2 = 0;
  for (a = 0; a < n; a++) {
    ab += occurrence[a] * probability[a];
    a2 += occurrence[a] * occurrence[a];
    b2 += probability[a] * probability[a];
  }
  return 1 - (ab / (sqrt(a2) * sqrt(b2)));
}

void CompareJob() {
  FILE *prob_f, *occur_f;
  long long a, b, d;
  long long ct = 0;
  float bhattacharyya, cosine;
  float bestp[N];
  long long besto[N];
  long long bestpw[N], bestow[N];
  // char *bestpw[N], *bestow[N];
  time_t now, start = clock();

  prob_f = fopen(prob_file, "rb");
  fscanf(prob_f, "%lld ", &a);
  if (a != vocab.vocab_size) {
    printf("ERROR: mismatched vocab size\n");
    return;
  }
  occur_f = fopen(occur_file, "rb");
  fscanf(occur_f, "%lld ", &a);
  if (a != vocab.vocab_size) {
    printf("ERROR: mismatched vocab size\n");
    return;
  }
  real *probability = (real *)calloc(vocab.vocab_size, sizeof(real));
  long long *occurrence =
      (long long *)calloc(vocab.vocab_size, sizeof(long long));
  // for (a = 0; a < N; a++) {
  //   bestpw[a] = (char *)malloc(MAX_STRING * sizeof(char));
  //   bestow[a] = (char *)malloc(MAX_STRING * sizeof(char));
  // }

  while (1) {
    char w[MAX_STRING], w2[MAX_STRING];
    long long sum = 0;
    pt::ptree tree_node;

    if (feof(prob_f)) {
      printf("eof\n");
      break;
    }
    if (ct >= vocab.vocab_size) {
      printf("\nAll vocab read, while file not ended, the remaining of prob "
             "file is: ");
      while (!feof(prob_f)) {
        char ch = fgetc(prob_f);
        printf("%#08X ", ch);
      }
      printf(", and the remaining of occur file is: ");
      while (!feof(occur_f)) {
        char ch = fgetc(occur_f);
        printf("%#08X ", ch);
      }
      printf("\n");
      break;
    }

    for (a = 0; a < N; a++) {
      bestp[a] = 0;
      besto[a] = 0;
      bestpw[a] = 0;
      bestow[a] = 0;
      // bestpw[a][0] = 0;
      // bestow[a][0] = 0;
    }
    GetWord(w, prob_f);
    GetWord(w2, occur_f);

    std::string word(w);
    std::string word2(w2);

    if (word != word2) {
      printf("WTF: Words from two files are different: %s vs %s\n", w, w2);
      break;
    }

    int vocab_idx = SearchVocab(vocab, w);
    if (vocab_idx < 0) {
      printf("WTF: word '%s' not in vocab, \n", w);
      continue;
    }

    for (a = 0; a < vocab.vocab_size; a++) {
      fread(&probability[a], sizeof(float), 1, prob_f);
      for (b = 0; b < N; b++) {
        if (probability[a] > bestp[b]) {
          for (d = N - 1; d > b; d--) {
            bestp[d] = bestp[d - 1];
            // strcpy(bestpw[d], bestpw[d - 1]);
            bestpw[d] = bestpw[d - 1];
          }
          bestp[b] = probability[a];
          // strcpy(bestpw[b], vocab.words[a].word);
          bestpw[b] = a;
          break;
        }
      }

      fread(&occurrence[a], sizeof(long long), 1, occur_f);
      for (b = 0; b < N; b++) {
        if (occurrence[a] > besto[b]) {
          for (d = N - 1; d > b; d--) {
            besto[d] = besto[d - 1];
            // strcpy(bestow[d], bestow[d - 1]);
            bestow[d] = bestow[d - 1];
          }
          besto[b] = occurrence[a];
          // strcpy(bestow[b], vocab.words[a].word);
          bestow[b] = a;
          break;
        }
      }
      sum += occurrence[a];
    }

    pt::ptree bestp_node;
    for (a = 0; a < N; a++) {
      if (bestp[a] > 0) {
        pt::ptree value_node, v1, v2;
        v1.put_value(bestp[a]);
        v2.put_value(vocab.words[bestpw[a]].cn);
        value_node.push_back(std::make_pair("", v1));
        value_node.push_back(std::make_pair("", v2));
        bestp_node.add_child(vocab.words[bestpw[a]].word, value_node);
      }
    }
    pt::ptree besto_node;
    for (a = 0; a < N; a++) {
      if (besto[a] > 0) {
        pt::ptree value_node, v1, v2;
        v1.put_value(besto[a]);
        v2.put_value(vocab.words[bestow[a]].cn);
        value_node.push_back(std::make_pair("", v1));
        value_node.push_back(std::make_pair("", v2));
        besto_node.add_child(vocab.words[bestow[a]].word, value_node);
      }
    }

    bhattacharyya =
        BhattacharyyaDistance(probability, occurrence, vocab.vocab_size);
    cosine = CosineDistance(probability, occurrence, vocab.vocab_size);

    tree_node.put("good_occurrence", vocab.words[vocab_idx].gcn);
    tree_node.put("bad_occurrence", vocab.words[vocab_idx].bcn);
    tree_node.put("bhattacharyya", bhattacharyya);
    tree_node.put("cosine", cosine);
    if (!boost::property_tree::json_parser::verify_json(bestp_node, 0)) {
      printf("\n[%s] ignoring bestp_node:\n", w);
      for (a = 0; a < N; a++) {
        printf("%lld\t%f\n", bestpw[a], bestp[a]);
      }
    } else {
      tree_node.add_child("most_probable", bestp_node);
    }
    if (!boost::property_tree::json_parser::verify_json(besto_node, 0)) {
      printf("\n[%s] ignoring besto_node:\n", w);
      for (a = 0; a < N; a++) {
        printf("%lld\t%lld\n", bestow[a], besto[a]);
      }
    } else {
      tree_node.add_child("most_occurred", besto_node);
    }
    tree_node.put("cosine", cosine);

    if (!boost::property_tree::json_parser::verify_json(tree_node, 0)) {
      printf("\nsomething wrong here, the current word is: %s(%lld)\n", w, ct);
    } else {
      root.add_child(word, tree_node);
    }
    now = clock();
    ct++;
    if (ct == vocab.vocab_size || ct % 10 == 0) {
      printf("%cProcessed_words: %lld Progress: %.2f%%  "
             "Words/sec: %.2f",
             13, ct, ct / (real)(vocab.vocab_size) * 100,
             ct / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC));
    }
  }
  fclose(prob_f);
  fclose(occur_f);
}

void Compare() {
  FILE *prob_f;
  std::ofstream ofs;

  if (access(occur_file, F_OK) == -1) {
    printf("Error: occurrence file '%s' does not exist\n", occur_file);
    return;
  }

  prob_f = fopen(prob_file, "rb");
  if (prob_f == NULL) {
    printf("Error: probability file '%s' does not exist\n", prob_file);
    return;
  }

  ofs.open(output_file, std::ofstream::out);
  if (ofs.fail()) {
    printf("Error: could not write to output file '%s'\n", output_file);
    return;
  }
  CompareJob();
  printf("outputing...\n");
  pt::write_json(ofs, root);
  ofs.close();
  printf("output wrote to '%s' \n", output_file);
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
  if (argc == 1) {
    printf("WORD2VECTOR: compare the two distribution v 0.1c\n\n");
    printf("Options:\n");
    printf("\t-prob <file>\n");
    printf("\t\tprobability file\n");
    printf("\t-occur <file>\n");
    printf("\t\toccurrence file\n");
    printf("\t-vocab <file>\n");
    printf("\t\tvocabulary <file>\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 20)\n");
    return 0;
  }
  output_file[0] = 0;
  prob_file[0] = 0;
  occur_file[0] = 0;
  vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-prob", argc, argv)) > 0) {
    strcpy(prob_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-occur", argc, argv)) > 0) {
    strcpy(occur_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-vocab", argc, argv)) > 0)
    strcpy(vocab_file, argv[i + 1]);
  if (num_threads <= 0) {
    printf("number of threads must be larger than 0");
    return 1;
  }
  if (vocab_file[0] != 0) {
    ReadVocab(vocab, vocab_file);
    printf("read vocab file: vocab size is %lld\n", vocab.vocab_size);
  } else {
    printf("missing -vocab parameter\n");
    return 1;
  }
  Compare();
  return 0;
}
