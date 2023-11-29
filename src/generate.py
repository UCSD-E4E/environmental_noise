import numpy as np
import torch
import os
import math

def cosine(A, f, fs, x, phi = 0):
    return A*np.cos(2*math.pi*x*f/fs + phi)

def enviro_noise_numpy( clip_length, sr, segment_length, deployment):
    """
    Final stage of generating random noise in a numpy array. Assumes that you have already generated the median and standard deviation arrays
    ##TODO Add in the feature to pass a path to custom median and std dev array##
    
    """
    assert isinstance(clip_length,int)
    assert isinstance(sr, int)
    assert isinstance(segment_length, int)
    assert isinstance(deployment, str)
    
    assert (clip_length > 0) and (sr > 0) and (sr <= 48000) and (segment_length > 0)
    deployment_set = {"mdd", "scripps"}
    assert deployment in deployment_set
    
    median_vec = None
    std_dev_vec = None
    
    if deployment == "mdd":
        median_vec = np.load("parameters/peru_medians.npy")
        std_dev_vec = np.load("parameters/peru_std_devs.npy")
    elif deployment == "scripps":
        median_vec = np.load("parameters/scripps_medians.npy")
        std_dev_vec = np.load("parameters/scripps_std_devs.npy")
    
    
    output_clip = np.zeros((clip_length,))
    # Nyquist-Shannon Sampling Theorem
    max_freq = sr//2
    # Since the range we used was up to 24 kHz, we used 24000 ffts, which gave us 12001 frequency bins, so I divided by
    # two to make up for that here.
    fft_range = max_freq//2
    
    segment_count = int(clip_length/segment_length)
    step_arr = np.arange(segment_length)
    
    for freq in range(fft_range):
        for segment in range(segment_count):
            seg_start = segment_length * segment
            seg_end = segment_length * (segment+1)
            
            # building up sinusoidal wave parameters
            freq_1 = freq * 2
            freq_2 = freq_1 + 1
            
            phi_1 = np.random.rand(1) * 2 * math.pi
            phi_2 = np.random.rand(1) * 2 * math.pi

            A1 = np.random.randn(1) * std_dev_vec[freq] + median_vec[freq]
            A2 = np.random.randn(1) * std_dev_vec[freq] + median_vec[freq]
            
            output_clip[seg_start:seg_end] += cosine(A1, freq_1, sr, step_arr, phi_1) + cosine(A2, freq_2, sr, step_arr, phi_2)
            
    return output_clip

# sanity check
'''
TODO, turn into a unit test with git actions
import librosa
import soundfile as sf
output_path = "/mnt/passive-acoustic-biodiversity/environmental_noise/scripps_examples"
example_clip_path = "../../environmental_noise_old/Average_Peru_Audio_48kHz.WAV"
example_clip, sr = librosa.load(example_clip_path,sr=48000)
print(len(example_clip))
segment_length = sr
noise = enviro_noise_numpy(len(example_clip)//24, sr, segment_length, deployment="mdd")
sf.write("test.wav", noise, samplerate=sr)
'''
