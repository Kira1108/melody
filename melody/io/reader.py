from dataclasses import dataclass
import librosa
import numpy as np
import time
from typing import Generator
import math

@dataclass
class SimpleAudioReader:
    
    target_sr: int | None = 16000
    
    def read(self, fp:str) -> tuple[np.ndarray, int]:
        """
        Reads audio data from a file.
        Args:
            fp (str): The file path to read audio data from.
        Returns:
            audio (np.ndarray): The audio data read from the file.
            sampling_rate (int): The sampling rate of the audio data.
        """
        audio, sampling_rate = librosa.load(fp, sr=None)
        
        if self.target_sr and sampling_rate != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.target_sr)
            sampling_rate = self.target_sr
            
        return audio, sampling_rate
    
    def stream(self, fp:str, chunk_size:int=1024, wait:bool = True):
        """
        Streams audio data from a file.
        Args:
            fp (str): The file path to stream audio data from.
            chunk_size (int): The size of each chunk to read from the file. Defaults to 1024.
        Yields:
            audio_chunk (np.ndarray): A chunk of audio data read from the file.
        """
        audio, _ = self.read(fp)
        
        chunk_seconds = chunk_size / self.target_sr
        num_chunks = math.ceil(len(audio) / chunk_size)
        # num_chunks = len(audio) // chunk_size + 1
        for i in range(0, num_chunks):
            if wait:
                time.sleep(chunk_seconds)
            start = i * chunk_size
            end = start + chunk_size
            
            is_final = i == num_chunks - 1
            yield audio[start:end], is_final
            
    def generate(self, fp:str, chunk_size:int = 1024):
        """
        Generates audio data from a file.
        Args:
            fp (str): The file path to generate audio data from.
            chunk_size (int): The size of each chunk to read from the file. Defaults to 1024.
        Yields:
            audio_chunk (np.ndarray): A chunk of audio data read from the file.
        """
        return self.stream(fp, chunk_size, wait = False)


