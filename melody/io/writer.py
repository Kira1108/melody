import wave
from dataclasses import dataclass
from typing import List

@dataclass
class BytesWavWriter:
    """
    A class to write audio data to a WAV file.
    Attributes:
        fp (str): The file path to write the WAV file to. Defaults to 'output.wav'.
        num_channels (int): The number of audio channels. Defaults to 1.
        sample_width (int): The sample width in bytes. Defaults to 2.
        frame_rate (int): The frame rate (samples per second). Defaults to 16000.
    Methods:
        __post_init__():
            Initializes the file path if not provided.
        write(data: List[bytes] | bytes) -> None:
            Writes the provided audio data to the WAV file.
            Args:
                data (List[bytes] | bytes): The audio data to write. Can be a list of bytes or a single bytes object.
    """
    fp: str = None
    num_channels: int = 1
    sample_width: int = 2
    frame_rate: int = 16000
    
    def __post_init__(self):
        if not self.fp:
            self.fp = 'output.wav'
    
    def write(self, data:List[bytes] | bytes) -> None:
        
        if isinstance(data, bytes):
            data = [data]
        
        with wave.open(self.fp, 'wb') as wav_file:
            num_frames = len(data) // self.sample_width
            wav_file.setnchannels(self.num_channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.frame_rate)
            wav_file.setnframes(num_frames)
            wav_file.writeframes(b''.join(data))
            
        print(f"Audio data written to {self.fp}")