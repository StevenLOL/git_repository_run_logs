https://github.com/bastibe/SoundFile.git


```
import IPython.display
import soundfile as sf

x, fs = sf.read(trainFiles[3],channels=1,samplerate=16000,format='RAW',subtype='PCM_16')

# Play it back!
IPython.display.Audio(data=x, rate=fs)

```
