# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:01:17 2020

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
from scipy.fftpack import dct

def power_signal(frames, NFFT=512):
    '''
    We can now do an N-point FFT on each frame to calculate the frequency spectrum, which is also called Short-Time Fourier-Transform (STFT), where N is typically 256 or 512, NFFT = 512; and then compute the power spectrum (periodogram) using the following equation:

    P=|FFT(xi)|2N
    
    where, xi is the ith frame of signal x.
'''
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames

######################################################################################################    
    
def filter_banks(pow_frames, sample_rate, nfilt=40, low_freq_mel=0):
    
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    return filter_banks

######################################################################################################

def mfcc(filter_banks, nu_ceps = 13):
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    return mfcc

######################################################################################################
    
def mean_normalization(features, axis=0):
    features -= (np.mean(features, axis=axis) + 1e-8)
    return features

######################################################################################################
    
