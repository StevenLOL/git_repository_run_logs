modeldir=/tmpdata/deepspeech_model/models
./deepspeech $modeldir/output_graph.pb $1 $modeldir/alphabet.txt $modeldir/lm.binary $modeldir/trie
