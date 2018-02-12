CC = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -std=c++11

all: cv cotrain compare predict vocab word2vec word2phrase distance word-analogy compute-accuracy

cv : compare-vectors.cpp
	$(CC) compare-vectors.cpp -o compare-vectors.o $(CFLAGS)
cotrain : cotrain.cpp
	$(CC) cotrain.cpp -o cotrain.o $(CFLAGS)
compare : compare.cpp
	$(CC) compare.cpp -o compare.o $(CFLAGS)
predict : predict.cpp
	$(CC) predict.cpp -o predict.o $(CFLAGS)
vocab : buildvocab.cpp
	$(CC) buildvocab.cpp -o buildvocab.o $(CFLAGS)
word2vec : word2vec.cpp
	$(CC) word2vec.cpp -o word2vec.o $(CFLAGS)
word2phrase : word2phrase.cpp
	$(CC) word2phrase.cpp -o word2phrase.o $(CFLAGS)
distance : distance.cpp
	$(CC) distance.cpp -o distance.o $(CFLAGS)
word-analogy : word-analogy.cpp
	$(CC) word-analogy.cpp -o word-analogy.o $(CFLAGS)
compute-accuracy : compute-accuracy.cpp
	$(CC) compute-accuracy.cpp -o compute-accuracy.o $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf *.o
