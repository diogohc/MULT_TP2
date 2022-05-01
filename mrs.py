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
from scipy.spatial import distance
import csv


warnings.filterwarnings("ignore")


dim_mfcc = 13
num_stats = 7


filesPath = "./dataset/all/"
files = os.listdir(filesPath)
numFiles = len(files)


#flags
NORMALIZAR_100_FEATURES = False
EXTRAIR_FEATURES = False
NORMALIZAR_FEATURES_EXTRAIDAS = False
CALCULAR_DISTANCIAS_100_FEATURES = False
CALCULAR_DISTANCIAS_FEATURES_EXTRAIDAS = False
CRIAR_RANKING = True


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
        if (maximo-minimo == 0):
            m_normalizada[:,i] = 0 # estou na duvida se é 0 ou 1
        else:
            m_normalizada[:,i] = (m[:,i]-minimo) / (maximo-minimo)
    return m_normalizada


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


def extrair_mfcc_e_calcular_stats(y):

    mfcc= librosa.feature.mfcc(y, n_mfcc = dim_mfcc)
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



def extrair_spec_centroid_e_calcular_stats(y):
    sc = librosa.feature.spectral_centroid(y)
    sc_stats = calcular_estatisticas(sc)
    return sc_stats


def extrair_spec_bandwith_e_calcular_stats(y):
    spec_bw = librosa.feature.spectral_bandwidth(y)
    spec_bw_stats = calcular_estatisticas(spec_bw)
    return spec_bw_stats
    

def extrair_spec_contrast_e_calcular_stats(y):
    spec_cont=librosa.feature.spectral_contrast(y)
    spec_cont_stats=calcular_estatisticas(spec_cont)
    return spec_cont_stats


def extrair_spec_flatness_e_calcular_stats(y):
    spec_flat = librosa.feature.spectral_flatness(y)
    spec_flat_stats=calcular_estatisticas(spec_flat)
    return spec_flat_stats


def extrair_spec_rolloff_e_calcular_stats(y):
    spec_roll = librosa.feature.spectral_rolloff(y)
    spec_roll_stats=calcular_estatisticas(spec_roll)
    return spec_roll_stats


def extrair_freq_fundamental_e_calcular_stats(y, fs):

    freq_fund = librosa.yin(y, fmin=20, fmax=fs/2)


    freq_fund[freq_fund==max(freq_fund)]=0
    
    freq_fund_stats = calcular_estatisticas(freq_fund)
    return freq_fund_stats


def extrair_rms_e_calcular_stats(y):
    rms = librosa.feature.rms(y)
    
    rms_stats = calcular_estatisticas(rms)
    return rms_stats


def extrair_zcr_e_calcular_stats(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    
    zcr_stats = calcular_estatisticas(zcr)
    return zcr_stats


def extrair_tempo(y): 
    
    tempo= librosa.beat.tempo(y)
    return tempo



def extrair_features():
    features_extraidas=[190*[0]]
    features_extraidas=np.asarray(features_extraidas)

    for i in range(numFiles):
        nome_ficheiro=filesPath+files[i]
        y, fs = librosa.load(nome_ficheiro)
        arr=[]
        mfcc = extrair_mfcc_e_calcular_stats(y)
        
        scent = extrair_spec_centroid_e_calcular_stats(y)
        
        sband = extrair_spec_bandwith_e_calcular_stats(y)
        
        scont = extrair_spec_contrast_e_calcular_stats(y)
        
        sflat = extrair_spec_flatness_e_calcular_stats(y)
        
        sroll = extrair_spec_rolloff_e_calcular_stats(y)
        
        freq_fund = extrair_freq_fundamental_e_calcular_stats(y, fs)
        
        rms = extrair_rms_e_calcular_stats(y)
        
        zcr = extrair_zcr_e_calcular_stats(y)
        
        tempo = extrair_tempo(y)
        
    
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
    


def distancia_euclidiana(m):
    m_dif=np.zeros((m.shape[0],m.shape[0]))
    
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if(i==j):
                break
            dist = np.linalg.norm(m[i,:] - m[j,:])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif


def distancia_manhattan(m):
    m_dif=np.zeros((m.shape[0],m.shape[0]))
    
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if(i==j):
                break
            dist = distance.cityblock(m[i,:] ,m[j,:])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif


def distancia_cosseno(m):
    m_dif=np.zeros((m.shape[0],m.shape[0]))
    
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if(i==j):
                break
            dist = distance.cosine(m[i,:] ,m[j,:])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif



def tratar_linha(nome_query, linha):
    ranking_musicas=np.array([nome_query])
    
    indices_para_ordenacao = np.argsort(linha)
    
    ranking20 = indices_para_ordenacao[1:21]
    
    ranking_ord_dec = np.flip(ranking20)
    
    for indice in ranking_ord_dec:
        ranking_musicas = np.append(ranking_musicas, files[indice])
    
    return ranking_musicas



def cria_ranking(query1, query2, query3, query4, m_distancias):
    mranking=np.array(["."]*21)
    array_querys = [query1, query2, query3, query4]
    
    while(array_querys):
        for i in range(len(files)):
            if(files[i] in array_querys):
                mranking = np.vstack([mranking, tratar_linha(files[i], m_distancias[i])])
                array_querys.remove(files[i])
    return mranking


if __name__ == "__main__":
    plt.close('all')
    
    #Normalizar features do ficheiro "top100_features"-------------------------------------------------
    if(NORMALIZAR_100_FEATURES):
        matriz = ler_fich_features("./ficheiros/top100_features.csv")
        print(matriz.shape)
        m = normalizar_features(matriz)
        escrever_em_ficheiro_csv("./ficheiros/top100_features_normalizadas.csv",m)


    if(EXTRAIR_FEATURES):
        extrair_features()
    
    #Normalizar features extraidas
    if(NORMALIZAR_FEATURES_EXTRAIDAS):
        matriz_features_extraidas= np.genfromtxt("./ficheiros/features_extraidas.csv", delimiter=",")
        mfeatures=normalizar_features(matriz_features_extraidas)
        escrever_em_ficheiro_csv("./ficheiros/features_normalizadas.csv",mfeatures)
    #--------------------------------------------------------------------------------------------
    
    
    #CALCULAR DISTANCIAS-------------------------------------------------------------------
    #features extraidas
    if(CALCULAR_DISTANCIAS_FEATURES_EXTRAIDAS):
        m_features = np.genfromtxt("./ficheiros/features_normalizadas.csv", delimiter=",")
    
        m_dist_euc = distancia_euclidiana(m_features)
        m_dist_man = distancia_manhattan(m_features)
        m_dist_cos = distancia_cosseno(m_features)

    
        escrever_em_ficheiro_csv("./ficheiros/dist_euclidiana_features_extraidas.csv", m_dist_euc)
        escrever_em_ficheiro_csv("./ficheiros/dist_manhattan_features_extraidas.csv", m_dist_man)
        escrever_em_ficheiro_csv("./ficheiros/dist_cosseno_features_extraidas.csv", m_dist_cos)

    
    #top 100 features
    if(CALCULAR_DISTANCIAS_100_FEATURES):
        m_100_features = np.genfromtxt("./ficheiros/top100_features_normalizadas.csv", delimiter=",")
        
        m100_dist_euc = distancia_euclidiana(m_100_features)
        m100_dist_man = distancia_manhattan(m_100_features)
        m100_dist_cos = distancia_cosseno(m_100_features)
        
        escrever_em_ficheiro_csv("./ficheiros/dist_euclidiana_100_features.csv", m100_dist_euc)
        escrever_em_ficheiro_csv("./ficheiros/dist_manhattan_100_features.csv", m100_dist_man)
        escrever_em_ficheiro_csv("./ficheiros/dist_cosseno_100_features.csv", m100_dist_cos)
    #--------------------------------------------------------------------------------------------
        
        
        
    #Criar ranking
    if(CRIAR_RANKING):
        q1 = "MT0000202045.mp3"
        q2 = "MT0000379144.mp3"
        q3 = "MT0000414517.mp3"
        q4 = "MT0000956340.mp3"
        
        #top 100 features
        #euclidiana
        dist_100_euc = np.genfromtxt("./ficheiros/dist_euclidiana_100_features.csv", delimiter=",")
        m_ranking_100_euc = cria_ranking(q1, q2, q3, q4, dist_100_euc)
        np.savetxt("./ficheiros/rankings/ranking_100_features_euclidiana.csv", m_ranking_100_euc[1:], 
                   delimiter=',',fmt='%s')
        
        #manhattan
        dist_100_man = np.genfromtxt("./ficheiros/dist_manhattan_100_features.csv", delimiter=",")
        m_ranking_100_man = cria_ranking(q1, q2, q3, q4, dist_100_man)
        np.savetxt("./ficheiros/rankings/ranking_100_features_manhattan.csv", m_ranking_100_man[1:], 
                   delimiter=',',fmt='%s')
    
    
        #cosseno
        dist_100_cos = np.genfromtxt("./ficheiros/dist_cosseno_100_features.csv", delimiter=",")
        m_ranking_100_cos = cria_ranking(q1, q2, q3, q4, dist_100_cos)
        np.savetxt("./ficheiros/rankings/ranking_100_features_cosseno.csv", m_ranking_100_cos[1:], 
                   delimiter=',',fmt='%s')
    
    
        #features extraidas
        #euclidiana
        dist_100_euc = np.genfromtxt("./ficheiros/dist_euclidiana_features_extraidas.csv", delimiter=",")
        m_ranking_100_euc = cria_ranking(q1, q2, q3, q4, dist_100_euc)
        np.savetxt("./ficheiros/rankings/ranking_features_extraidas__euclidiana.csv", m_ranking_100_euc[1:], 
                   delimiter=',',fmt='%s')
        
        #manhattan
        dist_100_man = np.genfromtxt("./ficheiros/dist_manhattan_features_extraidas.csv", delimiter=",")
        m_ranking_100_man = cria_ranking(q1, q2, q3, q4, dist_100_man)
        np.savetxt("./ficheiros/rankings/ranking_features_extraidas_manhattan.csv", m_ranking_100_man[1:], 
                   delimiter=',',fmt='%s')
    
    
        #cosseno
        dist_100_cos = np.genfromtxt("./ficheiros/dist_cosseno_features_extraidas.csv", delimiter=",")
        m_ranking_100_cos = cria_ranking(q1, q2, q3, q4, dist_100_cos)
        np.savetxt("./ficheiros/rankings/ranking_features_extraidas_cosseno.csv", m_ranking_100_cos[1:], 
                   delimiter=',',fmt='%s')
        
        