# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:34:38 2020

@author: SauravChavan, VipulYadav, ShravaniPattewar, ShrutiSankpal

Citation:
@misc{fayek2016,
            title   = "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between",
            author  = "Haytham M. Fayek",
            year    = "2016",
            url     = "https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"
            }
"""

import numpy as np

def pre_empasize(signal, alpha=1, cof=0.98):
    '''
    In speech processing, the original signal usually has too much lower frequency energy, 
    and processing the signal to emphasize higher frequency energy is necessary. 
    To perform pre-emphasis, we choose some value α between .9 and 1. 
    Then each value in the signal is re-evaluated using this formula: y[n] = x[n] - α*x[n-1]. 
    This is apparently a first order high pass filter.
    Source: https://math.stackexchange.com/
            https://github.com/astorfi/speechpy/blob/master/speechpy/processing.py
    '''
    rolled_signal = np.roll(signal,shift)
    return signal - alpha*rolled_signal

######################################################################################################

def framing(emphasized_signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    '''Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) 
    overlap between consecutive frames. Popular settings are 25 ms for the frame size, 
    frame_size = 0.025 and a 10 ms stride (15 ms overlap), frame_stride = 0.01.
    '''
    frame_length, frame_step = frame_size * rate, frame_stride * rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating   any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]
    print("The total number of frames:",frame_length)
    
    return frames, frame_length

######################################################################################################
    
def hamming(frames, frame_length):
    '''
    After slicing the signal into frames, we apply a window function such as the Hamming window to each frame. A Hamming window has the following form:

    w[n]=0.54−0.46cos(2πnN−1)
    '''
    frames *= np.hamming(frame_length)
    
    return frames

######################################################################################################
