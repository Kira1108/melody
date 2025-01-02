import logging
logging.basicConfig(level=logging.INFO)

from melody.io.reader import SimpleAudioReader
from melody.asr.paraformer import Paraformer
from melody.vad.fsmn import Vad

audio_fp = "./datafiles/recording.wav"

reader = SimpleAudioReader()
asr = Paraformer.auto(seconds = 0.6) # 这个chunk可以设置的稍微小一点
vad = Vad(chunk_size = 200) # 这里vad不能设置的太大了，这样就不好折腾了, 这里的200和paraformer的chunksize没啥关系

# the chunk size is defined by the asr model, 0.6 seconds corresponds to 9600 frames
for chunk in reader.stream(
    audio_fp, 
    chunk_size = asr.chunk_stride):
    
    transcription = asr.stream_asr(chunk)
    print(transcription)
    
    if vad.shutup(chunk):
        print("*"*30, "Shut up detected", "*"*30)
        
print("*"*30, "Shut up detected", "*"*30)