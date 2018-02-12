make word2vec
make vocab

if [ $# -lt 3 ]; then
  echo "Usage: ./train.sh <good_file> <bad_file> <outdir> <training_options...>"
  exit
fi

GOOD="$1"
BAD="$2"
OUTDIR="/u/kanyuan/sbout/jargon/word2vec/jargon/$3"

if [ ! -e "$GOOD" ]; then
  printf "Good file '%s' does not exist\n" $GOOD
  exit
fi

if [ ! -e "$BAD" ]; then
  printf "Bad file '%s' does not exist\n" $BAD
  exit
fi

if [ ! -d "$GOOD" ]; then
  mkdir -p "$OUTDIR"
fi

VOCAB=$OUTDIR"goodbad.vocab"
if [ ! -e "$VOCAB" ]; then
  printf "\nVocab file not found, learning vocab from texts\n"
  ./buildvocab.o -good $GOOD -bad $BAD -output $VOCAB
fi
if [ ! -e "$VOCAB" ]; then
  printf "\n Error: cannot create vocab file...\n"
  exit
fi
printf "Vocab file loaded at '%s'\n" $VOCAB

TRAIN_FLAGS="${@:4}"
# "-cbow 0 -size 300 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 30 -binary 1 -iter 30"

GOOD_OUT=$OUTDIR"good_vectors.bin"
printf "\n=======training good texts======\n"
printf "Output file: %s\n" $GOOD_OUT
time ./word2vec.o -train $GOOD -output $GOOD_OUT -read-vocab $VOCAB $TRAIN_FLAGS
printf "=======training good texts finished======\n"

BAD_OUT=$OUTDIR"bad_vectors.bin"
printf "\n=======training bad texts======\n"
printf "Output file: %s\n" $BAD_OUT
time ./word2vec.o -train $BAD -output $BAD_OUT -read-vocab $VOCAB $TRAIN_FLAGS
printf "=======training bad texts finished======\n"
