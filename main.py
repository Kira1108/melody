import logging
logging.basicConfig(level=logging.INFO)

from melody.io.reader import SimpleAudioReader
from melody.asr.paraformer import Paraformer
from melody.vad.fsmn import FMSNVad
from melody.puncs.ct_trans import PuncCreator
from melody.nlu.turn_pred import TurnDetector

def gen_transcription(
    audio_fp:str = "./datafiles/recording.wav", 
    wait:bool = False):

    reader = SimpleAudioReader()
    asr = Paraformer.auto(seconds = 0.6) # 这个chunk可以设置的稍微小一点
    vad = FMSNVad(chunk_size = 200) # 这里vad不能设置的太大了，这样就不好折腾了, 这里的200和paraformer的chunksize没啥关系
    punc = PuncCreator()
    td = TurnDetector()


    buffer = ''
    # the chunk size is defined by the asr model, 0.6 seconds corresponds to 9600 frames
    for chunk, is_final in reader.stream(
        audio_fp, 
        chunk_size = asr.chunk_stride, wait=wait):
        
        # asr inference
        transcription = asr.stream_asr(chunk)[0]['text']
        buffer += transcription
        status = "transient"
        
        # 句尾截止
        if is_final:
            status = "stable"
            content = punc(buffer)
            buffer = ''
            val = {
                "transcription": transcription,
                "status":status,
                "content":content
            }
            
        # 语意和音频截止
        elif vad.shutup(chunk) and td.shutup(buffer):
            status = "stable"
            content = punc(buffer)
            buffer = ''
            val = {
                "transcription": transcription,
                "status":status,
                "content":content
            }
            
        # 未截止
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