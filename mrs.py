#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa #https://librosa.org/
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st

warnings.filterwarnings("ignore")

sampleRate = 22050
window_length = 92.88
useMono = True
frame_length = 92.88 
ms = 23.22
hop_length = 23.22
dim_mfcc = 13
num_stats = 7

def ler_fich_features():
    nome_fich = "./ficheiros/top100_features.csv"

    data = np.genfromtxt(nome_fich, delimiter=",")

    #apagar primeira linha
    data=np.delete(data, 0, 0)
    #apagar primeira coluna
    data = np.delete(data, 0, 1)
    #apagar ultima coluna
    data = np.delete(data, data.shape[1]-1, 1)

    return data
    
    
    
    
def normalizar_features(m):
    m_normalizada = m
    nl,nc=m_normalizada.shape
    for i in range(nc):
        maximo = m[:,i].max()
        minimo = m[:,i].min()
        m_normalizada[:,i] = (m[:,i]-minimo) / (maximo-minimo)
    return m_normalizada


def escrever_em_ficheiro_csv(matriz):
    np.savetxt("./ficheiros/top100_features_normalizadas.csv", matriz)
    
def calcular_estatisticas(array):
    array=array.flatten()
    media = np.mean(array)
    stdev = np.std(array)
    skewness = st.skew(array)
    kurtosis = st.kurtosis(array)
    median = np.median(array)
    maximo = np.max(array)
    minimo = np.min(array)
    array_stats = np.array([media,stdev,skewness,kurtosis,median,maximo,minimo])
    return array_stats


def extrair_mfcc_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]

    mfcc= librosa.feature.mfcc(fich, sr= sampleRate, n_mfcc = dim_mfcc)
    nl, nc = mfcc.shape
    mfcc_stats=np.zeros((nl,num_stats))
    #calcular estatisticas
    for i in range(nl):
        media = np.mean(mfcc[i,:])
        stdev = np.std(mfcc[i,:])
        skewness = st.skew(mfcc[i,:])
        kurtosis = st.kurtosis(mfcc[i,:])
        median = np.median(mfcc[i:,])
        maximo = np.max(mfcc[i:,])
        minimo = np.min(mfcc[i:,])
        mfcc_stats[i:,] = np.array([media, stdev, skewness, kurtosis, median, maximo, minimo])
        #mfcc_stats[i:,] = calcular_estatisticas(mfcc[i,:])
    mfcc_stats=mfcc_stats.flatten()
    return mfcc_stats




def extrair_spec_centroid_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    sc = librosa.feature.spectral_centroid(fich, sr=sampleRate, hop_length=hop_length, 
                                           win_length=window_length)
    sc_stats = calcular_estatisticas(sc)
    return sc_stats


def extrair_spec_bandwith_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    spec_bw = librosa.feature.spectral_bandwidth(fich, sr=sampleRate, hop_length=hop_length, 
                                           win_length=window_length)
    spec_bw_stats = calcular_estatisticas(spec_bw)
    return spec_bw_stats
    

if __name__ == "__main__":
    plt.close('all')
    """
    #--- Load file
    fName = "Queries/MT0000414517.mp3"
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)
    
    #--- Play Sound
    #sd.play(y, sr, blocking=False)
    
    #--- Plot sound waveform
    plt.figure()
    #librosa.display.waveplot(y)
    
    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
    #--- Extract features    
    rms = librosa.feature.rms(y = y)
    rms = rms[0, :]
    print(rms.shape)
    times = librosa.times_like(rms)
    plt.figure(), plt.plot(times, rms)
    plt.xlabel('Time (s)')
    plt.title('RMS')
    """
    
    matriz = ler_fich_features()
    m = normalizar_features(matriz)
    escrever_em_ficheiro_csv(m)
    
    filesPath = "./dataset/all/"
    files = os.listdir(filesPath)
    numFiles = len(files)
    extrair_mfcc_e_calcular_stats(filesPath+files[0])
    #extrair_spectral_centroid_e_calcular_stats(filesPath+files[0])
    



    