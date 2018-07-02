# GFCC
The source codes of Gammatone features for noise-robust speech recognition.
https://github.com/uwjunqi/Gfcc
## install

src/install.sh
## usage

### prepare input

Generate input text file ([fileA](https://github.com/uwjunqi/Gfcc/blob/master/egs/code.scp)) with following format:
```
file1.wav file1.gfcc
file2.wav file2.gfcc
```
### generate gfcc features
```
Gfcc_extract fileA
```

### read gfcc features

The output is in HTK format (see bool feat_gfcc::WriteFtrDat(const std::string FeatFn, const std::string WavFn) in jun-featGfcc.cc)

To read HTK features in python:

Download [HTK.py](https://github.com/danijel3/PyHTK/blob/master/python/HTK.py)

```
from HTK import HTKFile
htk = HTKFile()
htk.load('file1.gfcc')
print len(htk.data)
print len(htk.data[0])
```
The feature is in N*24 
