# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:24:28 2022

@author: 35196
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

#remover a média do sinal
def remove_mean(emg_mV,time):
    emg_correctmean = emg_mV - np.mean(emg_mV)
    plt.plot(time, emg_correctmean)
    plt.grid()
    plt.xlabel('Tempo (s)')
    plt.ylabel('EMG (mV)')
    return emg_correctmean

#cálculo do rms do sinal de 3 formas
def rolling_rms(x, N):
    xc = np.cumsum(abs(x)**2);
    return np.sqrt((xc[N:] - xc[:-N])/ N)

def window_rms(a, window_size=2):
    return np.sqrt(sum([a[window_size-i-1:len(a)-i]**2 for i in range(window_size-1)])/window_size)

def window_rms1(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'valid'))

#retificação do sinal
def emg_rectified(emg_correctmean, time):
    # process EMG signal: rectify
    emg_rectified = abs(emg_correctmean)
    # plot comparison of unrectified vs rectified EMG
    plt.figure(figsize=(8,6))
    plt.plot(time, emg_correctmean)
    plt.grid()
    plt.xlabel(' Tempo / s ')
    plt.ylabel(' Amplitude / mV ')
    plt.yticks([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0, 0.1 , 0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    plt.xticks([0,10,20,30,40,50,60,70,80])
    plt.tick_params(labelsize=14)
    return emg_rectified

#envelope do sinal pela aplicação de um filtro passa baixo
def envelope(emg_rectified,time):
    low_pass= 3
    sfreq = 1000
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/(sfreq/2)
    b2, a2 = signal.butter(1, low_pass, btype='lowpass')
    emg_envelope = signal.filtfilt(b2, a2,emg_rectified )
    fig = plt.figure()
    fig.set_size_inches(w=9,h=4)
    # plt.plot(time,emg_rectified)
    plt.plot(time, emg_envelope)
    plt.xlabel('Tempo (s)') 
    plt.ylabel('EMG(mv)')
    return emg_envelope

#plot do sinal original
def raw_emg(emg,time):
    plt.plot(time,emg)
    plt.xlabel('Tempo(s)')
    plt.ylabel('Raw EMG')
    plt.grid()
    plt.show()

#plot do sinal em mV
def mv_emg(emg_correctmean,time):
    plt.plot(time,emg_correctmean)
    plt.xlabel('Tempo(s)')
    plt.ylabel('EMG(mV)')
    plt.grid()
    plt.show()
    
#envelope rms
def envelope_rms(emg_correctmean,time,time2):
    rms = window_rms1(emg_correctmean,1000)
    plt.plot(time,emg_correctmean,color = 'b')
    plt.plot(time2,rms)
    plt.xlabel('Tempo / s')
    plt.ylabel('EMG / mv')
    return rms    
  
#aplicação de um filtro aos 50 Hz
def notch_filter(emg_correctmean,sr,time): 
    #quero tirar 50 e as suas harmonicas
    notch_freq = 50 
    quality_factor = 5.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sr) 
    freq, h = signal.freqz(b_notch, a_notch, fs= sr) 
    emg_notchfilter = signal.filtfilt(b_notch, a_notch, emg_correctmean)
    fig = plt.figure()
    fig.set_size_inches(w=9,h=4)
    plt.plot(time,emg_correctmean)
    plt.xlabel('Tempo(s)') 
    plt.ylabel('EMG(mv)')
    fig = plt.figure()
    plt.title('sem 50')
    plt.plot(time,emg_notchfilter)
    plt.xlabel('Tempo(s)') 
    plt.ylabel('EMG(mv)')
    return emg_notchfilter

#dividir o sinal pelas 5 contrações  
def janelas(emg_correctmean,time,sr):
    contraction_segments = list()
    t_start=7*sr
    t_end=18*sr
    while t_end*sr < len(time)*sr - 7*sr:
       contraction_segments.append(emg_correctmean[t_start:t_end])
       t_start +=15*sr
       t_end += 15*sr
    return contraction_segments

#dividir o sinal pelas 5 contrações  
def contraction_windows(emg_correctmean,onsets,offsets,time,sr):
    contraction_segments = list()
    n=0
    while n<len(offsets):
        t_start= onsets[n]
        #print(t_start)
        t_end=offsets[n]
        if(len(emg_correctmean[t_start:t_end])>0.5*sr):
            contraction_segments.append(emg_correctmean[t_start:t_end])  
        n+=1
    #fig = plt.figure()
    #fig.set_size_inches(w=9,h=4)
    #plt.plot(time[t_start:t_end],contraction_segments[n-1])
    #plt.xlabel('Tempo(s)') 
    #plt.ylabel('EMG(mv)')    
    return contraction_segments

def rest_windows(emg_correctmean,onsets,offsets,time,sr):
    rest_segments = list()
    n=0
    t_start= 0
    while n<len(offsets):
        t_end=(onsets[n]-sr)
        if(len(emg_correctmean[t_start:t_end])>0.5*sr):
            rest_segments.append(emg_correctmean[t_start:t_end])  
        t_start=(offsets[n]+sr)
        n+=1     
    rest_segments.append(emg_correctmean[t_start:len(emg_correctmean)])
    #fig = plt.figure()
    #fig.set_size_inches(w=9,h=4)
    #plt.plot(time[t_start:len(emg_correctmean)],rest_segments[n])
    #plt.xlabel('Tempo(s)') 
    #plt.ylabel('EMG(mv)')    
    return rest_segments


#FFT do sinal
def frequencia(emg_correctmean,title,sr,time, f_ratio=1):
    ft = np.fft.fft(emg_correctmean)
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

#PSD (densidade espetral de potência) de cada medição
def welch(emg_correctmean,sr):
    f, Pxx_den = signal.welch(emg_correctmean,sr,scaling='density',nperseg=1024)
    plt.figure()
    #plt.plot(f,Pxx_den)
    plt.semilogy(f,Pxx_den)
    plt.xlabel('frequência [Hz]')
    plt.ylabel('PSD [mV**2/Hz]')
    plt.show()
    return f, Pxx_den


#PSD da média das 11 medições
def welch_total(sinais, nomes, sr):
    plt.figure(1)
    plt.xlabel('frequência [Hz]')
    plt.ylabel('PSD [mV**2/Hz]')
    pxx=[]
    media=[]
    soma=0
    for nome, sinal in zip(nomes, sinais):
        f, Pxx_den = signal.welch(sinal,sr,scaling='density',nperseg=1024)
        pxx.append(Pxx_den)
        plt.semilogy(f, Pxx_den, label=nome)
    for i in range(len(pxx[0])):
        for p in pxx:
            soma += p[i]
        media.append(soma / float(len(pxx)))
        soma=0
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.semilogy(f, media)
    plt.xlabel('frequência [Hz]')
    plt.ylabel('PSD [mV**2/Hz]')
    plt.title('PSD '+ nomes[0][:-11])
    plt.show()
    return media

#envelope RMS da média das 11 medições  
def envelope_rms_total(envelope_RMS,nomes,tempo,sr):
    plt.figure()
    plt.xlabel('tempo [s]')
    plt.ylabel('Amplitude [mV]')
    media_envelope=[]
    des_padrao_envelope=[]
    soma2=np.array([])
    comprimentos=np.array([])
    for nome, envelop, temp in zip(nomes, envelope_RMS,tempo):      
        plt.plot(temp, envelop, label=nome)
        comprimentos = np.append(comprimentos, [len(envelop)])
    for i in range(int(np.amin(comprimentos))):
        for v in envelope_RMS:
            soma2 = np.append(soma2, [v[i]])
        media_envelope.append(np.mean(soma2))
        des_padrao_envelope.append(np.std(soma2))
        soma2=np.array([])          
    plt.legend()
    plt.show()
    plt.figure()
    time = np.array([i/sr for i in range(0, len(media_envelope), 1)])
    plt.plot(time,media_envelope)
    plt.xlabel('tempo [s]')
    plt.ylabel('Amplitude [mV]')
    plt.title('Envelope RMS '+ nomes[0][:-11])
    plt.show()
    return media_envelope, des_padrao_envelope
    

def new_csv_file():
    
    
    return 0



 



    



