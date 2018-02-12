./buildvocab.o -good text8_nonstop -bad text8_nonstop -output text8_nonstop.vocab
echo
echo "word2vec"
echo
time ./word2vec.o -train text8_nonstop -output text8_nonstop.vectors -cbow 0 -size 200 -window 10 -negative 25 -hs 0 -read-vocab text8_nonstop.vocab -sample 1e-4 -threads 20 -binary 0 -iter 15
echo
echo "prediction"
echo
