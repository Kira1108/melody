from pydub import AudioSegment
import librosa
import numpy as np

def convert_sr_file(
    input_file:str, 
    output_file:str,
    target_sr=8000):
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_frame_rate(target_sr)
    audio.export(output_file, format="wav")
    print("Converted audio file to target sampling rate, saved to: ", output_file)
    
    
def convert_sr_numpy(
    audio:np.ndarray, 
    orig_sr:int, 
    target_sr:int):
    
    if isinstance(audio, list):
        audio = np.array(audio)
        
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)