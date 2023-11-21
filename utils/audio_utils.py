import scipy.signal as sci_signal
import os
import librosa
from typing import List, Tuple
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
def read_wav(path: str):
    y, sr = librosa.load(path,mono=True)
    return y, sr

def write_wav(signal : List[float],fs : int,path: str):
    sf.write(
        file = path,
        data=signal,
        samplerate=fs,
        format="WAV"
    )
    
def write_freq_params (path,params) : 
    params = np.array(params,dtype=np.float32)
    with open(path,"wb") as f : 
        f.write(params.tobytes())
        
def read_freq_params (path):
    with open(path,"rb") as f :
        params = np.fromfile(f,dtype=np.float32)    
    return params
    
def write_raw(signal, fs, path):
    signal = np.array(signal, dtype=np.float32)
    with open(path, "wb") as f:
        # Escribe la frecuencia de muestreo (fs) como bytes de 4 bytes
        f.write(fs.to_bytes(4, byteorder="little"))
        # Escribe la señal como bytes
        f.write(signal.tobytes())


def read_raw(path):
    with open(path, "rb") as f:
        # Lee la frecuencia de muestreo (fs) desde el archivo
        fs_bytes = f.read(4)
        fs = int.from_bytes(fs_bytes, byteorder="little")
        
        # Lee la señal desde el archivo en el formato float32
        signal = np.fromfile(f, dtype=np.float32)

    return signal, fs

def get_mfcc (signal,fs:int,n_mfcc = 15) :
    mfcc_signal = librosa.feature.mfcc(
        y=signal,
        sr=fs,
        n_mfcc=n_mfcc
    )
    
    return mfcc_signal


def get_beats(y, sr, hop_length: int = 512):
    _, frames = librosa.beat.beat_track(sr=sr, y=y, hop_length=hop_length)
    beat_times = librosa.frames_to_samples(frames=frames, hop_length=hop_length)
    return beat_times

def split(signal,fs : int, seconds: float = 0.1):
    split_length = int(seconds * fs)
    
    signal_segments = []
    for split_point in range(0, len(signal) - split_length, split_length):
        segment = signal[split_point: split_point + split_length]
        signal_segments.append(segment)
        
    return signal_segments

def combine(signal_set):
    num_signals = len(signal_set)
    
    if num_signals == 0:
        raise ValueError("No signal sets provided to combine.")
    
    min_length = min([len(signal) for signal, _ in signal_set])
    combined_signal = np.zeros(min_length)
    frame_samples = signal_set[0][1]  # Tomamos la frecuencia de muestreo del primer conjunto
    
    for i in range(0, num_signals):
        signal, fs = signal_set[i]
        
        if len(signal) != min_length:
            raise ValueError(f"Signal in set {i} has a different length. All signals must have the same length.")
        
        combined_signal += signal
        
    return combined_signal, frame_samples

def time_normalizer (path : str,time : float = 1.5,write = True) :
    signal,fs = read_wav(path)
    target_length = int(time * fs)
    signal_length = len(signal)    
    if signal_length > target_length : 
        norm_signal = signal[:target_length]
    
    elif signal_length < target_length : 
        missing_length = target_length - signal_length
        norm_signal = np.concatenate((signal,np.zeros(missing_length)))
        
    else :
        norm_signal = signal
    
    if(write == True):
        write_wav(
            signal=norm_signal,
            fs=fs,
            path=path
        )
    
    return norm_signal

def freq_spec(signal, fs):
    spec = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(spec), 1 / fs)
    mags = np.abs(spec)

    max_magnitude_index = np.argmax(mags)
    max_magnitude_frequency = round(freqs[max_magnitude_index], 2)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mags)
    plt.scatter(freqs[max_magnitude_index], mags[max_magnitude_index], color='red', marker='o', label=f'Max Magnitude Frequency: {max_magnitude_frequency} Hz')
    plt.title("Señal")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.xlim(0, 2000)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def event_split (signal,fs,t_group_tolerance = 0.1) :
    
    group_tolerance = t_group_tolerance * fs
    peaks,_ = sci_signal.find_peaks(signal,height=0.05)
    
    start_idxs = []
    for i in range(len(peaks)-1) : 
        peak = peaks[i]
        if peak == peaks[0] : 
            start_idxs.append(peak)
        else : 
            if peaks[i] - peaks[i - 1] >= group_tolerance : 
                start_idxs.append(peak)
                
    fragments = [signal[start_idxs[i] : start_idxs[i + 1]] for i in range(len(start_idxs) - 1)]
    
    return fragments
            
def compress(signal,fs) : 
    new_fs = int(5512.5)
    compress_factor = int(fs / new_fs)
    comp_signal = sci_signal.decimate(signal,compress_factor)
    
    return comp_signal,new_fs

def aprox_to_notes (arr) :
    with open("assets/notes.json") as jsonf : 
        note_dict : dict = json.load(jsonf)["inverse_note_freqs"]
        notes = np.array(list((note_dict.keys())),dtype=float)
        aproxs = [notes[np.argmin(np.abs(notes - num))] for num in arr]
    return aproxs

class Param_generator () : 
    def __init__ (self) : 
        self.param_layout = self.load_param_layout()
            
    def load_param_layout (self) :
        with open("assets/notes.json") as jsonf : 
            param_layout = json.load(jsonf)["inverse_note_freqs"]
            for key in param_layout : 
                param_layout[key] = 0
                
            return param_layout
        
    def get_freq_param_set (self,sample_set : list,print_progress=True,return_max_sample_amps = False) :
                
        if print_progress == True :
            sample_set = tqdm(sample_set)      
               
        result_params = []
        max_sample_amps = []
        
        for signal,fs in sample_set :     
            #obtener amplitud maxima del fragmento de audio
            if return_max_sample_amps : 
                max_amp = np.max(np.abs(signal))
                max_sample_amps.append(max_amp)
              
            np.set_printoptions(suppress=True)
            spec = np.fft.fft(signal)
            freqs = np.abs(np.fft.fftfreq(len(spec), 1/fs))
            mags = np.abs(spec)
            
            half_len = int(len(freqs) // 2)
            
            #frecuencias
            freqs = freqs[:half_len]
            aprox_freqs = np.array(aprox_to_notes(freqs))
            #magnituddes
            mags = np.array(mags[:half_len])
            max_mag = np.max(mags)
            round_mags = np.round(np.divide(mags,max_mag),4)
            
            #sumar magnitud de duplicados en una malla de frecuencias
            stacks = np.column_stack((aprox_freqs,round_mags))
            unique_stacks = np.unique(stacks,axis=0)

            unprocessed_param_mesh : dict = dict(self.param_layout)
            for freq,mag in unique_stacks : 
                if mag > 0 and mag < 4200 :
                    unprocessed_param_mesh[str(freq)] += mag
            
            #formatear y normalizar resultados
            freqs = np.array(list(unprocessed_param_mesh.keys()),dtype=np.float32)
            mags = np.array(list((unprocessed_param_mesh.values())),dtype=np.float32)
            max_mag = np.max(mags)
            norm_mags = np.round(np.divide(mags,max_mag),4)
            processed_param_mesh = np.column_stack((freqs,norm_mags))
            
            result_params.append(processed_param_mesh)
            
        result_params = np.array(result_params,dtype=np.float32)
        
        if return_max_sample_amps == True : 
            return result_params,max_sample_amps
        
        if return_max_sample_amps == False :
            return result_params
         
    


        
    

    

    
 
    

    
  
    
    
    
    

    
    
        