# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:45:59 2024

@author: tppin
"""

from numpy import array, linspace
from numpy import loadtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EMGfunctions as emgf
from scipy import signal
import csv
import os
from tqdm import tqdm
#import tsfel
import pandas as pd
import neurokit2 as nk


path = 'Tests/'
file_name = path[:-1]
save_path = r'C:\Users\tppin\Documents\CENIMAT\Papers\Ongoing\LIG_ECGElectrodes\EMG_signals\Python' #diretoria da pasta
path_id = next(os.walk(path))[2]
file_name_path = os.path.join(save_path, file_name + '.csv')

channel_column = 5
# taxa de amostragem
sr = 1000
# resolução do Bitalino
resolution = 10
# Ganho do sensor
G_emg = 1009
# tensão operacional
Vcc = 3.3
window_size = 1000
t_start = 7*sr
t_end = 18*sr
sinais = []
nomes = []
tempo = []
envelope_RMS = []
média_amplitudes = []
number_of_files = 0
contraction_onsets = pd.DataFrame()
contraction_offsets = pd.DataFrame()
csv_file = open(file_name_path, 'w')
dados_writer = csv.writer(csv_file)  

for n, id_ in tqdm(enumerate(path_id), total=len(path_id)):
    # abrir o ficheiro emg a analisar
    line = np.array([])
    data = loadtxt(path + id_)
    emg = data[:, channel_column]
    
    
    # função de transferência para passar valores ADC para mV
    emg_mv = ((((emg/2**resolution)-(1/2)) * Vcc)/G_emg)*1000
    emg_correctmean = emg_mv-np.mean(emg_mv)
    print(np.max(emg_correctmean))
    
    # vetor tempo com taxa de amostragem de 1000 Hz
    time = np.array([i/sr for i in range(0, len(emg_correctmean), 1)])
    df_merge_data=pd.DataFrame({'Time':time, 'Raw':emg_mv, 'Correct mean':emg_correctmean})
    
    #filtered = emgf.notch_filter(emg_correctmean, sr, time)
    #rectificação de sinal e envelope"
    emg_rectified = emgf.emg_rectified(emg_correctmean, time)
    emg_envelope = emgf.envelope(emg_rectified, time)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Envelope':emg_envelope})], axis = 1)
    
    #deteção de activações
    activity, info = nk.emg_activation(emg_amplitude=emg_envelope, method="mixture")
    
    #nk.events_plot([info["EMG_Onsets"], info["EMG_Offsets"]], emg_envelope)
    #contraction_onsets_list = pd.concat([Contraction_onsets,pd.DataFrame(info["EMG_Onsets"])],axis=1)
    #contraction_offsets_list = pd.concat([Contraction_offsets,pd.DataFrame(info["EMG_Offsets"])],axis=1)
    #Contraction_onsets.insert(n,n,info["EMG_Onsets"])
    #Contraction_offsets.insert(n,n,info["EMG_Offsets"])
    emg_signals, info = nk.emg_process(emg_rectified, sr)
   
    #nk.emg_plot(emg_signals, info)
    # tempo rmsenvelope
    contraction_onsets=info["EMG_Onsets"]
    contraction_offsets=info["EMG_Offsets"]
    start = int(window_size/2)
    end = int(len(time) - window_size/2 + 1)
    time2 = time[start:end]
    # envelope rms
    rmsenvelope = emgf.envelope_rms(emg_correctmean, time, time2)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'RMS Envelope':rmsenvelope})], axis = 1)
    #segmentação de contrações
    contraction_segments = list()
    contraction_segments = emgf.contraction_windows(rmsenvelope,contraction_onsets,contraction_offsets, time, sr)
    rest_segments = list()
    rest_segments = emgf.rest_windows(rmsenvelope,contraction_onsets,contraction_offsets, time, sr)
    snr = np.array([])
    amplitudes = np.array([])
    ruido= np.array([])
    amplitudes_max = np.array([])
    for i in range(len(contraction_segments)):
        ruido_i= np.mean(rest_segments[i])  
        ruido= np.append(ruido, ruido_i)
        snr_i = 20 * \
            np.log10(
                np.mean(contraction_segments[i])/ruido[i])
        print(snr_i)
        
        #line = np.append(line, ruido_i)
        amplitudes_i = np.mean(contraction_segments[i])
        amplitudes = np.append(amplitudes, amplitudes_i)
        line = np.append(line, amplitudes_i)
        snr = np.append(snr, snr_i)
        line = np.append(line, snr_i)
        amplitudes_max_i = np.max(contraction_segments[i])
        amplitudes_max = np.append(amplitudes_max, amplitudes_max_i)
        line = np.append(line, amplitudes_max_i)
        
    freq, emg_fft = emgf.frequencia(emg_correctmean, "FFT"+id_, sr, time, f_ratio=0.5)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Frequencies':freq})], axis = 1)  
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'FFT':emg_fft})], axis = 1) 
    
    freq_bin, psd = emgf.welch(emg_correctmean, sr)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Frequency_bins':freq_bin})], axis = 1)  
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'PSD':psd})], axis = 1) 
    
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'SNR list':snr})], axis = 1)
    
    
    #write csv file with data from Pandas dataframe with merged data for Origin processing
    df_merge_data.to_csv(os.path.join(save_path, path, id_ + '.csv'), index=False)
    
csv_file.flush()
print("Ficheiro Terminado")
csv_file.close()      
      
#print(number_of_files)
    
    



