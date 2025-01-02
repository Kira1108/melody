from functools import lru_cache

from funasr import AutoModel

from melody.utils.log import timer


@lru_cache(maxsize = None)
def load_punc_model(stream = False):
    if not stream:
        return AutoModel(model="ct-punc")
    return AutoModel(model="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")

class PuncCreator:
    """Add punctuations to Chinese text with FunASR models."""
    def __init__(self):
        self.model = load_punc_model(stream = False)
        
    @timer(name = "Punctuation generation")
    def create_punc(self, text: str):
        return self.model.generate(input=text)[0]['text']
    
    def __call__(self, text: str):
        return self.create_punc(text)
    
    
class StreamPuncCreator:
    """Add punctuations to Chinese text with FunASR models."""
    def __init__(self):
        self.model = load_punc_model(stream = True)
        self.cache = {}
        
    @timer(name = "Stream Punctuation generation")
    def create_punc(self, text: str):
        return self.model.generate(input=text, cache = self.cache)[0]['text']
    
    def __call__(self, text: str):
        return self.create_punc(text)
    
if __name__ == "__main__":
    text = "我是一个好人"
    punc_creator = PuncCreator()
    print(punc_creator(text)) 