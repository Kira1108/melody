from dataclasses import dataclass, field
from functools import lru_cache
from typing import Union

import numpy as np
from funasr import AutoModel

from melody.utils.log import timer

"""
The Paraformer model is a transformer-based model that can perform streaming ASR implemented in the funasr library.(Alibaba)
"""

@lru_cache
def load_model():
    return AutoModel(model="paraformer-zh-streaming")

@dataclass
class Paraformer:
    """
    A class to perform streaming ASR using the Paraformer model.
    Note: for each audio, you need to create a new instance of this class, because the cache is not shared between instances.
    
    """
    chunk_size: list = field(default_factory=lambda: [0, 10, 5])
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    
    def __post_init__(self):
        self.model = load_model()
        self.cache = {}
        
    @classmethod
    def auto(cls, seconds = 0.6):
        """根据语音时长自动设置chunk_size"""
        f = int(seconds * 1000 / 60)
        chunk_size = [0,f,int(f/2)]
        return cls(chunk_size = chunk_size)
    
    @property
    def chunk_stride(self):
        "返回chunk_stride，其实是有多少个frames"
        seconds = self.chunk_size[1] * 60 / 1000
        return int(seconds * 16000)
    
    @timer(name = "ParaformerStreaming")
    def stream_asr(self, 
                   speech_chunk:Union[list, np.array], 
                   is_final:bool = False):
        if isinstance(speech_chunk, list):
            speech_chunk = np.array(speech_chunk)
            
        return self.model.generate(
            input=speech_chunk, 
            cache=self.cache, 
            is_final=is_final, 
            chunk_size=self.chunk_size, 
            encoder_chunk_look_back=self.encoder_chunk_look_back, 
            decoder_chunk_look_back=self.decoder_chunk_look_back
        )
        

        