import numpy as np
import torch
torch.set_num_threads(1)
import pyaudio
from functools import lru_cache 
import threading

@lru_cache(maxsize=None)
def load_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
    return model, utils

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()
num_samples = 512
continue_recording = True

def stop():
    input("Press Enter to stop the recording:")
    global continue_recording
    continue_recording = False

def start_recording():
    model, utils = load_model()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    data = []
    voiced_confidences = []
    
    global continue_recording
    continue_recording = True    
    stop_listener = threading.Thread(target=stop)
    stop_listener.start()
    print("Start Recoding...")
    while continue_recording:
        
        audio_chunk = stream.read(num_samples)
        data.append(audio_chunk)
        audio_int16 = np.frombuffer(audio_chunk, np.int16);
        audio_float32 = int2float(audio_int16)
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)
        print(new_confidence, end='\r', flush=True)
        
if __name__ == "__main__":
    start_recording()
