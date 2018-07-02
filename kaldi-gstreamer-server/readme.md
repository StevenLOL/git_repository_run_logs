Real-time full-duplex speech recognition server, based on the Kaldi toolkit and the GStreamer framwork.

# Prerequisit
https://github.com/alumae/kaldi-gstreamer-server
## libs
```
 sudo apt install libgstreamer1.0-dev gstreamer1.0-plugins-good gstreamer1.0-tools gstreamer1.0-pulseaudio
 sudo pip install ws4py==0.3.2 tornado
```
## build kaldi
```
#build kaldi first then
#in kaldi/src

 make ext
 export GST_PLUGIN_PATH=~/tools/gst-kaldi-nnet2-online/src
```

Pretrained models are in test/models

#Usage
## start a master server

```
python ./kaldigstserver/master_server.py --port=8888
```
 
