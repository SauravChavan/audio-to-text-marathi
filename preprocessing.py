# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:34:38 2020

@author: SauravChavan, VipulYadav, ShravaniPattewar, ShrutiSankpal
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

def windowing(pre_empasized_signal):
    # add code for windowing signal
    
def power_spectrum(windowed_signal):
    # add code for getting power spectrum of the signal
