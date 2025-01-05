from silero_vad import load_silero_vad, get_speech_timestamps
import matplotlib.pyplot as plt
from melody.io.reader import SimpleAudioReader
from functools import lru_cache
from dataclasses import dataclass

@lru_cache(maxsize=None)
def load_model():
    return load_silero_vad()

@dataclass
class SileroOffline:
    
    def __post_init__(self):
        self.model = load_silero_vad()
        
    def vad(self, wav):
        return get_speech_timestamps(
            wav,
            self.model,
            return_seconds=True)
        
    def plot(self, wav, sr = 16000):
        speech_timestamps = self.vad(wav)
        seconds = len(wav) // sr + 1
        fig, ax = plt.subplots(figsize=(15, 3))
        for vad in speech_timestamps:
            plt.axvspan(
                vad['start'] * sr, 
                vad['end'] * sr, 
                color='skyblue', 
                alpha=0.5)
            
        for i in range(seconds):
            plt.axvline(i * sr, color='gray', 
                        alpha=0.8, lw = 0.5, ls = '--')
        
        plt.plot(wav, alpha=0.6, lw = 0.8)
        plt.title("Voice Activity Detection")
        plt.show()
        
        
if __name__ == "__main__":
    reader = SimpleAudioReader()
    vad = SileroOffline()
    wav, sr = reader.read('./datafiles/recording.wav')
    vad.plot(wav,sr)