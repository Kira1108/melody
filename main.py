import logging
logging.basicConfig(level=logging.INFO)

from melody.io.reader import SimpleAudioReader
from melody.asr.paraformer import Paraformer
from melody.vad.fsmn import FMSNVad
from melody.puncs.ct_trans import PuncCreator

def gen_transcription(audio_fp:str = "./datafiles/recording.wav"):

    reader = SimpleAudioReader()
    asr = Paraformer.auto(seconds = 0.6) # 这个chunk可以设置的稍微小一点
    vad = FMSNVad(chunk_size = 200) # 这里vad不能设置的太大了，这样就不好折腾了, 这里的200和paraformer的chunksize没啥关系
    punc = PuncCreator()


    buffer = ''
    # the chunk size is defined by the asr model, 0.6 seconds corresponds to 9600 frames
    for chunk, is_final in reader.stream(
        audio_fp, 
        chunk_size = asr.chunk_stride, wait=False):
        
        # asr inference
        transcription = asr.stream_asr(chunk)[0]['text']
        buffer += transcription
        status = "transient"
        
        # stable state logic
        if vad.shutup(chunk) or is_final:
            status = "stable"
            content = punc(buffer)
            buffer = ''
            val = {
                "transcription": transcription,
                "status":status,
                "content":content
            }
        else:
            val = {
                "transcription": transcription,
                "status":status,
                "content":""
            }
        yield val
        
        
if __name__ == "__main__":
    for transcription in gen_transcription():
        if transcription['status'] == "stable":
            print(transcription)