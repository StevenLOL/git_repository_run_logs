OpenSMIL speech feature extractor


# install
```
git clone https://github.com/naxingyu/opensmile
cd opensmile
./buildStandalone.sh
```
# FAQ:

How to extract features in batch model ?
```
# just use same output file name, the features will append to the orignial file

SMILExtract -C ./config/ComParE_2016.conf -I 00176480_nohash_0.wav -O ./temp.txt
SMILExtract -C ./config/ComParE_2016.conf -I 00176480_nohash_0.wav -O ./temp.txt
SMILExtract -C ./config/ComParE_2016.conf -I 00176480_nohash_0.wav -O ./temp.txt
SMILExtract -C ./config/ComParE_2016.conf -I 00176480_nohash_0.wav -O ./temp.txt
```
