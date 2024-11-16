# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:07:38 2024

@author: tppin
"""

from numpy import array, linspace
from numpy import loadtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import csv
import os
from tqdm import tqdm
# import tsfel
import pandas as pd
import neurokit2 as nk
import ECG_functions as ecgf


path = 'Tests/'
file_name = path[:-1]
save_path = r'C:\Users\tppin\Documents\CENIMAT\Papers\Ongoing\LIG_ECGElectrodes\ECG signals\Python' #diretoria da pasta
path_id = next(os.walk(path))[2]
file_name_path = os.path.join(save_path, file_name + '.csv')

channel_column = 1
# taxa de amostragem
sr = 1000
# resolução do Bitalino
resolution = 10
# Ganho do sensor
gain = 1100
# tensão operacional
Vcc = 3.3
window_size = 1000
t_start = 7*sr
t_end = 18*sr
#sinais = []
#nomes = []
#tempo = []
number_of_files = 0
csv_file = open(file_name_path, 'w')
dados_writer = csv.writer(csv_file)

for n, id_ in tqdm(enumerate(path_id), total=len(path_id)):
    # abrir o ficheiro ecg a analisar
    line = np.array([])
    data = loadtxt(path + id_)
        #retirar dados ecg das colunas do ficheiro txt    
    ecg = data
    #função de transferência de ADC para mV 
    ecg_mv = ((((ecg/2**resolution)-(1/2)) * Vcc)/gain)*1000
    #print(np.max(ecg_mv))
    #correção de velor médio
    ecg_correctmean = ecg_mv - np.mean(ecg_mv)
    
    time = np.array([i/sr for i in range(0, len(ecg_correctmean), 1)])
    df_merge_data=pd.DataFrame({'Time':time, 'Raw':ecg_mv, 'Correct mean':ecg_correctmean})
    
    #plt.plot(time[10000:15000], ecg_correctmean[10000:15000])
    #plt.grid()
    #plt.xlabel('Tempo (s)')
    #plt.ylabel('ECG (mV)')
    
    signals, info = nk.ecg_process(ecg_correctmean, sr)
    #nk.ecg_plot(signals, info)
    p_peak_onsets=info['ECG_P_Onsets']
    t_peak_offsets=info['ECG_T_Offsets']
    tp_deviation = ecgf.tp_deviations(ecg_correctmean, p_peak_onsets, t_peak_offsets)
    
    q_onsets = info['ECG_Q_Peaks']
    s_onsets = info['ECG_S_Peaks']
    qrs_durations = ecgf.qrs_duration(q_onsets, s_onsets)
    
    r_peaks=info['ECG_R_Peaks']
    s_peaks=info['ECG_S_Peaks']
    
    #qrs_amplitudes=ecgf.qrs_amplitude(ecg_correctmean,r_peaks,s_peaks)
        
    p_peaks=info['ECG_P_Peaks']
    snr_list = ecgf.snr_values(ecg_correctmean, p_peaks, r_peaks)
    """
    #q_peaks=info['ECG_Q_Peaks']
    
    s_peaks=info['ECG_S_Peaks']
    #t_peaks=info['ECG_T_Peaks']"""
    #HeartRate
    df_heart_rate = pd.DataFrame({'Heart Rate':signals['ECG_Rate']})
    
    
    
    qrs_epochs = nk.ecg_segment(signals['ECG_Clean'],info['ECG_R_Peaks'],sampling_rate=1000)
    snr_epochs = ecgf.qrs_epochs_snr(qrs_epochs)
     
    df_merge_qrs=pd.DataFrame({'Time':qrs_epochs[str(1)]['Index']})
    for i in qrs_epochs:
        df_merge_qrs=pd.concat([df_merge_qrs,qrs_epochs[i]['Signal']], axis = 1)
        #plt.plot(time[0:len(qrs_epochs[str(i)])], qrs_epochs[str(i)]['Signal'])
        #plt.grid()
        #plt.xlabel('Tempo (s)')
        #plt.ylabel('EMG (mV)')
        
        #print(i)
    #df_merge_qrs.insert('','')    
    df_merge_qrs=pd.concat([df_merge_qrs,pd.DataFrame({'QRS_durations':qrs_durations})], axis = 1)
        
    wiener_filtered = signal.wiener(ecg_correctmean, 20).tolist()
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Filtered':wiener_filtered})], axis = 1)
    
    plt.plot(time[0:2000], wiener_filtered[0:2000])
    plt.grid()
    plt.xlabel('Tempo (s)')
    plt.ylabel('ECG (mV)')
    noise = list()
    n=0
    while n<=(len(wiener_filtered)-1):
        noise.append(ecg_correctmean[n]-wiener_filtered[n])
        n+=1
    
    plt.plot(time[0:2000], noise[0:2000])
    plt.grid()
    plt.xlabel('Tempo (s)')
    plt.ylabel('ECG (mV)') 
    
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Noise':noise})], axis = 1)
   
    
    #FFT sinal ECG
    freq, ecg_fft = ecgf.frequencia(wiener_filtered, "FFT"+id_, sr, time, f_ratio=0.5)
    freq_n, noise_fft = ecgf.frequencia(noise, "FFT"+id_, sr, time, f_ratio=0.5)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Frequencies':freq,'FFT_Signal':ecg_fft,'FFT_Noise':noise_fft})], axis = 1)
   
    
    #SNR sinal
    snr_value=20*np.log10((max(wiener_filtered)-min(wiener_filtered))/(max(noise)-min(noise)))
    snr=list()
    snr.append(snr_value)
    #print(snr_value)
    print('Mean SNR - ' + str(np.mean(snr_list)))
    print('Deviation SNR - ' + str(np.std(snr_list)))
    
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'SNR':snr})], axis = 1)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'R-Peaks':r_peaks})], axis = 1) 
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'SNR values':snr_list})], axis = 1)
    
    #PSD signal
    freq_bin, psd_signal = ecgf.welch(wiener_filtered, sr)
    freq_bin_n, psd_noise = ecgf.welch(noise, sr)
    df_merge_data=pd.concat([df_merge_data,pd.DataFrame({'Freq_PSD':freq_bin,'PSD_Signal':psd_signal,'PSD_Noise':psd_noise})], axis = 1)
           
    """
    #Write csv file with signal segment and processing
    df_merge_qrs.to_csv(os.path.join(save_path, path, id_ + '_epochs.csv'), index=False)
    df_merge_data.to_csv(os.path.join(save_path, path, id_ + '.csv'), index=False)
    df_heart_rate.to_csv(os.path.join(save_path, path, id_ + 'heart_rate.csv'), index=False)"""