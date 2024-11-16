# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:00:49 2024

@author: tppin
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True

import neurokit2 as nk

def qrs_epochs(cleaned_signals, r_peaks, sr):
    
    
    return 0

def tp_deviations(signal, onsets, offsets):
    tp_deviation = list()
    
    #compute tp deviation
    m=0
    while m<len(onsets):
        start=offsets[m]
        end=onsets[m]
        if(isinstance(start, int) and isinstance(end, int)):
            deviation =np.sqrt(abs(signal[start]-signal[end]))
            #print(deviation)
            tp_deviation.append(deviation)
        m+=1
    
    return tp_deviation

def qrs_duration(q_peaks, s_peaks):
    qrs_durations = list()
    
    n=0
    while n<len(q_peaks):
        duration = (s_peaks[n]-q_peaks[n])/1000
        #print(duration)
        #if(isinstance(duration, int)):
        qrs_durations.append(duration)
        n+=1
    return qrs_durations

def qrs_amplitude(signal, r_peaks, s_peaks):
    qrs_amplitudes=list()
    n=0
    while n<len(r_peaks):
        index_s=r_peaks[n]
        index_e=s_peaks[n]
        amplitude = (signal[index_s]-signal[index_e])
        qrs_amplitudes.append(amplitude)
        n+=1
    
    return qrs_amplitudes

def qrs_epochs_snr(epochs):
    snr_epochs = list()
    for i in epochs:
        epoch=epochs[i]['Signal'].tolist()
        epoch_filtered = signal.wiener(epoch)
        epoch_noise = list()
        n=0        
        while n<(len(epoch_filtered)):
            epoch_noise.append(epoch[n]-epoch_filtered[n])
            n+=1
        snr_ep=10*np.log10((max(epoch_filtered)-min(epoch_filtered))/(max(epoch_noise)-min(epoch_noise)))
        #print(snr_ep)
        snr_epochs.append(snr_ep)
    return snr_epochs

def snr_values(signal_correctmean, p_peaks, r_peaks):
    snr_values = list()
    #print(len(p_peaks))
    p_peaks=[x for x in p_peaks if not np.isnan(x)]
    i=0
    while i<len(p_peaks)-1:
        start = p_peaks[i]
        end = p_peaks[i+1]
        segment = signal_correctmean[start:end]
        noise_segment = list()
        filtered_segment = signal.wiener(segment, 20).tolist()
        n=0
        while n<=(len(segment)-1):
            noise_segment.append(segment[n]-filtered_segment[n])
            n+=1
        
        #segmentvpp=(max(segment)-min(segment))
        snr_value=20*np.log10((max(segment)-min(segment))/(max(noise_segment)-min(noise_segment)))
        #print(snr_value)
        snr_values.append(snr_value)
        i+=1
    #snr_values.append(0)
    """
    #plot values of SNR vs. time during measures
    snr_values.append(0)
    plt.figure()
    plt.plot(r_peaks, snr_values)
    plt.grid()
    plt.xlabel('Tempo (s)')
    plt.ylabel('SNR (dB)')
    """
    return snr_values

def frequencia(ecg_correctmean,title,sr,time, f_ratio=1):
    ft = np.fft.fft(ecg_correctmean)
    #n= sr * np.max(time)
    magnitude= abs(ft)
    plt.figure(figsize=(18,5))
    frequency= np.linspace(0,sr,len(magnitude))
    n_bins= int(len(frequency) * f_ratio)
    magnitude = magnitude[:n_bins]
    #s_dbfs = 20 * np.log10(magnitude/(32768-1))
    plt.plot(frequency[:n_bins], magnitude)
    plt.xlabel("Frequência(Hz)")
    plt.ylabel("|FFT|")
    plt.title(title)
    return frequency[:n_bins],magnitude

def welch(ecg_correctmean,sr):
    f, Pxx_den = signal.welch(ecg_correctmean,sr,scaling='density',nperseg=1024)
    plt.figure()
    #plt.plot(f,Pxx_den)
    plt.semilogy(f,Pxx_den)
    plt.xlabel('frequência [Hz]')
    plt.ylabel('PSD [mV**2/Hz]')
    plt.show()
    return f, Pxx_den