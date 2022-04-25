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
WL = 92.88
useMono = True
frame_length = 92.88
ms = 23.22
hop_length = 23.22
dim_mfcc = 13
num_stats = 7


filesPath = "./dataset/all/"
files = os.listdir(filesPath)
numFiles = len(files)



def ler_fich_features(nome_fich):

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


def normalizar_vetor(v):
    v_normalizado = v
    for i in range(v.shape[0]):
        maximo = v.max()
        minimo = v.min()
        v_normalizado[i] = (v[i]-minimo) / (maximo-minimo)
    return v_normalizado


def escrever_em_ficheiro_csv(nome, matriz):
    np.savetxt(nome, matriz, delimiter=',')
    
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
    sc = librosa.feature.spectral_centroid(fich, sr=sampleRate)
    sc_stats = calcular_estatisticas(sc)
    return sc_stats


def extrair_spec_bandwith_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    spec_bw = librosa.feature.spectral_bandwidth(fich, sr=sampleRate)
    spec_bw_stats = calcular_estatisticas(spec_bw)
    return spec_bw_stats
    


def extrair_spec_contrast_e_calcular_stats(nome_ficheiro):
    fich= librosa.load(nome_ficheiro)[0]
    spec_cont=librosa.feature.spectral_contrast(fich, sr=sampleRate)
    spec_cont_stats=calcular_estatisticas(spec_cont)
    return spec_cont_stats


def extrair_spec_flatness_e_calcular_stats(nome_ficheiro):
    fich= librosa.load(nome_ficheiro)[0]
    spec_flat = librosa.feature.spectral_flatness(fich)
    spec_flat_stats=calcular_estatisticas(spec_flat)
    return spec_flat_stats


def extrair_spec_rolloff_e_calcular_stats(nome_ficheiro):
    fich= librosa.load(nome_ficheiro)[0]
    spec_roll = librosa.feature.spectral_rolloff(fich, sr=sampleRate)
    spec_roll_stats=calcular_estatisticas(spec_roll)
    return spec_roll_stats


def extrair_freq_fundamental_e_calcular_stats(nome_ficheiro):
    fich, fs= librosa.load(nome_ficheiro)

    freq_fund = librosa.yin(fich, sr=sampleRate, fmin=20, fmax=fs/2)


    freq_fund[freq_fund==max(freq_fund)]=0
    
    freq_fund_stats = calcular_estatisticas(freq_fund)
    return freq_fund_stats


def extrair_rms_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    rms = librosa.feature.rms(fich)
    
    rms_stats = calcular_estatisticas(rms)
    return rms_stats



def extrair_zcr_e_calcular_stats(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    zcr = librosa.feature.zero_crossing_rate(fich)
    
    zcr_stats = calcular_estatisticas(zcr)
    return zcr_stats


def extrair_tempo(nome_ficheiro):
    fich = librosa.load(nome_ficheiro)[0]
    
    tempo= librosa.beat.tempo(fich)
    return tempo



def extrair_features():
    features_extraidas=[148*[0]]
    features_extraidas=np.asarray(features_extraidas)

    for i in range(numFiles):
        arr=[]
        mfcc = extrair_mfcc_e_calcular_stats(filesPath+files[i])
        
        scent = extrair_spec_centroid_e_calcular_stats(filesPath+files[i])
        
        sband = extrair_spec_bandwith_e_calcular_stats(filesPath+files[i])
        
        scont = extrair_spec_contrast_e_calcular_stats(filesPath+files[i])
        
        sflat = extrair_spec_flatness_e_calcular_stats(filesPath+files[i])
        
        sroll = extrair_spec_rolloff_e_calcular_stats(filesPath+files[i])
        
        freq_fund = extrair_freq_fundamental_e_calcular_stats(filesPath+files[i])
        
        rms = extrair_rms_e_calcular_stats(filesPath+files[i])
        
        zcr = extrair_zcr_e_calcular_stats(filesPath+files[i])
        
        tempo = extrair_tempo(filesPath+files[i])
    
    
        arr=np.append(mfcc,scent)
        arr=np.append(arr,sband)
        arr=np.append(arr,scont)
        arr=np.append(arr,sflat)
        arr=np.append(arr,sroll)
        arr=np.append(arr,freq_fund)
        arr=np.append(arr, rms)
        arr=np.append(arr, zcr)
        arr=np.append(arr, tempo)
    
        features_extraidas=np.vstack([features_extraidas, arr])
        print(i)

    escrever_em_ficheiro_csv("./ficheiros/features_extraidas.csv",features_extraidas[1:])
    


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
    
    matriz = ler_fich_features("./ficheiros/top100_features.csv")
    print(matriz.shape)
    m = normalizar_features(matriz)
    escrever_em_ficheiro_csv("./ficheiros/top100_features_normalizadas.csv",m)
    

    #extrair_features()
    
    
    matriz_features_extraidas= np.genfromtxt("./ficheiros/features_extraidas.csv", delimiter=",")
    print(matriz_features_extraidas.shape)
    print(matriz_features_extraidas[0])
    mfeatures=normalizar_features(matriz_features_extraidas)
    escrever_em_ficheiro_csv("./ficheiros/features_normalizadas.csv",mfeatures)
    