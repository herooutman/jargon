#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "const.h"
#include "vocab.h"
#include "util.h"
#include "mathutil.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

typedef float real;
namespace pt = boost::property_tree;

char good_file[MAX_STRING], bad_file[MAX_STRING], vocab_file[MAX_STRING],
    output_file[MAX_STRING];
int debug_mode = 2, num_threads = 20, binary = 0;
long long layer1_size = -1;

real *syn0_good, *syn0_bad;
Vocab vocabulary;
clock_t start;

void Compare() {
  pt::ptree root;
  long long a, b;
  char *word;
  std::ofstream ofs;
  real cosine, hamming;
  layer1_size = read_vectors(good_file, vocabulary, &syn0_good, binary);
  a = read_vectors(bad_file, vocabulary, &syn0_bad, binary);
  if (a != layer1_size) {
    printf("ERROR: mismatched vector sizes.");
    exit(1);
  }

  ofs.open(output_file, std::ofstream::out);
  if (ofs.fail()) {
    printf("Error: could not write to output file '%s'\n", output_file);
    return;
  }

  real *v1 = (real *)malloc(layer1_size * sizeof(real));
  real *v2 = (real *)malloc(layer1_size * sizeof(real));
  for (a = 0; a < vocabulary.vocab_size; a++) {
    pt::ptree tree_node;
    word = vocabulary.words[a].word;
    for (b = 0; b < layer1_size; b++) {
      v1[b] = syn0_good[b + a * layer1_size];
      v2[b] = syn0_bad[b + a * layer1_size];
    }
    cosine = CosineDistance(v1, v2, layer1_size);
    hamming = HammingDistance(v1, v2, layer1_size);
    tree_node.put("hamming", hamming);
    tree_node.put("cosine", cosine);
    root.add_child(word, tree_node);
    // printf("Word: %s, Cosine: %.02f, Hamming: %.02f", word, cosine, hamming);
  }

  printf("outputing...\n");
  pt::write_json(ofs, root);
  ofs.close();
  printf("output wrote to '%s' \n", output_file);
}


// we are going to using trained vectors from bad corpus to predict in good
// corpus, so we use syn0 from bad and syn1 from good
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2VECTOR: prediction v 0.1c\n\n");
    printf("Options:\n");
    printf("\t-good <file>\n");
    printf("\t\tvector* (syn1) file trained from good corpus\n");
    printf("\t-bad <file>\n");
    printf("\t\tvector (syn0) file trained from bad corpus\n");
    printf("\t-output <file>\n");
    printf("\t\t<file> to store results\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 20)\n");
    printf("\t-vocab <file>\n");
    printf("\t\tvocab <file>\n");
    printf("\t-binary <int>\n");
    printf("\t\tLoad vectors with binary mode; default is 0 (off)\n");
    return 0;
  }
  output_file[0] = 0;
  vocab_file[0] = 0;
  good_file[0] = 0;
  bad_file[0] = 0;
  if ((i = ArgPos((char *)"-good", argc, argv)) > 0) {
    strcpy(good_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-bad", argc, argv)) > 0) {
    strcpy(bad_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-vocab", argc, argv)) > 0) {
    strcpy(vocab_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
    binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[i + 1]);

  if (num_threads <= 0) {
    printf("number of threads must be larger than 0");
    return 1;
  }

  if (vocab_file[0] != 0) {
    ReadVocab(vocabulary, vocab_file);
    printf("read vocab file: vocab size is %lld\n", vocabulary.vocab_size);
  } else {
    printf("missing -vocab parameter\n");
    return 1;
  }

  Compare();
  return 0;
}
