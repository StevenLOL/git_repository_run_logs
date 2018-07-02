# DeepSpeech

A TensorFlow implementation of Baidu's DeepSpeech architecture 

https://github.com/mozilla/DeepSpeech

# install

git clone --depth 1 https://github.com/mozilla/DeepSpeech.git

* only [cudnn6](https://developer.nvidia.com/rdp/cudnn-download) is supported!

# verify

## Getting the pre-trained model

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz

## ASR with pre-trained model

See [run_predict.sh](./run_predict.sh)

modeldir=/tmpdata/deepspeech_model/models
./deepspeech $modeldir/output_graph.pb [wav dir/file] $modeldir/alphabet.txt $modeldir/lm.binary $modeldir/trie

## ASR by file list

```
	printf("%s\n",argv[2]);
	std::ifstream infile(argv[2]);
	for( std::string line; getline( infile, line ); )
	{
		assert(sox_init() == SOX_SUCCESS);
		printf("%s\t",line.c_str());
	  sox_format_t* input = sox_open_read(line.c_str(), NULL, NULL, NULL);
		.
		.
		.
		.
	}
```
### make file

```
c++ -o deepspeech   `pkg-config --cflags sox` client.cc  -Wl,--no-as-needed -Wl,-rpath,. -L/development/DeepSpeech/native_client  -ldeepspeech -ldeepspeech_utils -ltensorflow_cc -ltensorflow_framework  `pkg-config --libs sox`


./native_client/deepspeech /tmpdata/deepspeech_model/models/output_graph.pb ./test.2.list /tmpdata/deepspeech_model/models/alphabet.txt
```

