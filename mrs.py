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
import time
import sys


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
CRIAR_RANKING_DIFERENCAS = False
CRIAR_RANKING_METADADOS = True
CALCULAR_PRECISAO = True
META_DATA = 1
OUVIR_4_2 = 1


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
    np.savetxt(nome, matriz, delimiter=',',fmt='%f')
    
    
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
        media = np.mean(mfcc[i])
        stdev = np.std(mfcc[i])
        skewness = st.skew(mfcc[i])
        kurtosis = st.kurtosis(mfcc[i])
        median = np.median(mfcc[i])
        maximo = np.max(mfcc[i])
        minimo = np.min(mfcc[i])
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
    nl, nc = spec_cont.shape
    spec_cont_stats=np.zeros((nl,num_stats))

    for i in range(nl):
        media = np.mean(spec_cont[i])
        stdev = np.std(spec_cont[i])
        skewness = st.skew(spec_cont[i])
        kurtosis = st.kurtosis(spec_cont[i])
        median = np.median(spec_cont[i])
        maximo = np.max(spec_cont[i])
        minimo = np.min(spec_cont[i])
        spec_cont_stats[i:,] = np.array([media, stdev, skewness, kurtosis, median, maximo, minimo])

    #spec_cont_stats=calcular_estatisticas(spec_cont)
    spec_cont_stats=spec_cont_stats.flatten()

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
        for j in range(i):
            dist = np.linalg.norm(m[i] - m[j])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif


def distancia_manhattan(m):
    m_dif=np.zeros((m.shape[0],m.shape[0]))
    
    for i in range(m.shape[0]):
        for j in range(i):
            dist = distance.cityblock(m[i] ,m[j])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif


def distancia_cosseno(m):
    m_dif=np.zeros((m.shape[0],m.shape[0]))
    
    for i in range(m.shape[0]):
        for j in range(i):
            dist = distance.cosine(m[i] ,m[j])
            m_dif[i][j]=m_dif[j][i]=dist
    return m_dif



def tratar_linha(nome_query, linha):
    ranking_musicas=np.array([nome_query])
    
    indices_para_ordenacao = np.argsort(linha)
    
    ranking20 = indices_para_ordenacao[1:21]
    
    
    for indice in ranking20:
        ranking_musicas = np.append(ranking_musicas, files[indice])
    
    return ranking_musicas



def cria_ranking(query1, query2, query3, query4, m_distancias):
    mranking=np.array(["."]*21)
    array_querys = [query1, query2, query3, query4]
    """
    while(array_querys):
        for i in range(len(files)):
            if(files[i] in array_querys):
                mranking = np.vstack([mranking, tratar_linha(files[i], m_distancias[i])])
                array_querys.remove(files[i])
    """     
    for query in array_querys:
        indice_query = files.index(query)
        mranking = np.vstack([mranking, tratar_linha(query, m_distancias[indice_query])])
        
    return mranking






def pontuacao_coluna_strings(string1, string2):
    pontos=0

    s1 = string1.split(';')
    s2 = string2.split(';')

    #limpar elementos (remover espacos finais e iniciais e ")
    for i in range(len(s1)):
        s1[i]=s1[i].strip()
        if(s1[i][0]=="\""):
            s1[i]=s1[i][1:]
        if(s1[i][-1]=="\""):
            s1[i]=s1[i][:-1]
    for i in range(len(s2)):
        s2[i]=s2[i].strip()
        if(s2[i][0]=="\""):
            s2[i]=s2[i][1:]
        if(s2[i][-1]=="\""):
            s2[i]=s2[i][:-1]
            
    #calcular similaridade
    for elem in s1:
        pontos += s2.count(elem)
 
    return pontos


def similaridade_metadados(m):
    msimilaridade = np.full((900,900), -1)
    for i in range(1, m.shape[0]):
        if(i==(900/2 + 1)):
            break
        for j in range(i,m.shape[0]):
            pontos=0
            #nao comparar a mesma musica
            if(m[i][0]==m[j][0]):
                continue
            
            #comparar artista
            if(m[i][1] == m[j][1]):
                pontos+=1
            #comparar quadrante
            if(m[i][3] == m[j][3]):
                pontos+=1
            #comparar moods
            pontos += pontuacao_coluna_strings(m[i][9], m[j][9])
            #comparar generos
            pontos += pontuacao_coluna_strings(m[i][11], m[j][11])
            
            msimilaridade[i-1][j-1] = msimilaridade[j-1][i-1] = pontos
    return msimilaridade




def tratar_linha_metadados(nome_query, linha):
    ranking_musicas=np.array(["."])
            
    indices_para_ordenacao = np.argsort(linha)
    
    ranking20 = indices_para_ordenacao[-20:]
    
    
    for indice in ranking20:
        ranking_musicas = np.append(ranking_musicas, files[indice])
    
    ranking_musicas=np.flip(ranking_musicas[1:])
    ranking_musicas = np.insert(ranking_musicas, 0, nome_query)
    return ranking_musicas


def criar_ranking_metadados(query1, query2, query3, query4, m_metadados):
    mranking=np.array(["."]*21)
    array_querys = [query1, query2, query3, query4]
                
    for query in array_querys:
        indice_query = files.index(query)
        mranking = np.vstack([mranking, tratar_linha_metadados(query, m_metadados[indice_query])])
    return mranking[1:]




def calcular_precisao(matriz_ranking_dif, matriz_ranking_metadados):
    mprecisao = np.array([0]*2)
    for i in range(matriz_ranking_dif.shape[0]):
        v_intersecao = np.intersect1d(matriz_ranking_dif[i][1:], matriz_ranking_metadados[i][1:])
        v_precisao = np.array([matriz_ranking_dif[i][0], v_intersecao.shape[0]/20])
        mprecisao = np.vstack([mprecisao, v_precisao])
    return mprecisao[1:]



def playing(query, ratingFile):
    
    if(os.path.exists(ratingFile) == False):
        print (ratingFile," nao existe crie o ficheiro ou exprimente outro")
        sys.exit()
    
    queriesList = os.listdir("./queries/")
    #print(queriesList)
    ind = queriesList.index(query)
    #(ind)
        
   
    playlist = np.genfromtxt(ratingFile, delimiter=",", dtype='unicode')
    #print(playlist)
    print(playlist[ind])
    for i in playlist[ind]:
        #print(type(i))
        music = filesPath+i
        print(music)

        warnings.filterwarnings("ignore")
        y, fs = librosa.load(music)
        
        #--- Play Sound
        #sd.play(y, sr, blocking=False)
        #time.sleep(10)
        
        
def main():
    #Normalizar features do ficheiro "top100_features"-------------------------------------------------
    if(os.path.exists("./ficheiros/top100_features_normalizadas.csv") == False):
        matriz = ler_fich_features("./ficheiros/top100_features.csv")
        print(matriz.shape)
        m = normalizar_features(matriz)
        escrever_em_ficheiro_csv("./ficheiros/top100_features_normalizadas.csv",m)


    if(os.path.exists("./ficheiros/features_extraidas.csv") == False):
        extrair_features()
    
    #Normalizar features extraidas
    if(os.path.exists("./ficheiros/features_normalizadas.csv") == False):
        matriz_features_extraidas= np.genfromtxt("./ficheiros/features_extraidas.csv", delimiter=",")
        mfeatures=normalizar_features(matriz_features_extraidas)
        escrever_em_ficheiro_csv("./ficheiros/features_normalizadas.csv",mfeatures)
    #--------------------------------------------------------------------------------------------
    
    
    #CALCULAR DISTANCIAS-------------------------------------------------------------------
    #features extraidas
    if(os.path.exists("./ficheiros/dist_euclidiana_features_extraidas.csv") == False and
        os.path.exists("./ficheiros/dist_manhattan_features_extraidas.csv") == False and
         os.path.exists("./ficheiros/dist_cosseno_features_extraidas.csv") == False):
        m_features = np.genfromtxt("./ficheiros/features_normalizadas.csv", delimiter=",")
    
        m_dist_euc = distancia_euclidiana(m_features)
        m_dist_man = distancia_manhattan(m_features)
        m_dist_cos = distancia_cosseno(m_features)

    
        escrever_em_ficheiro_csv("./ficheiros/dist_euclidiana_features_extraidas.csv", m_dist_euc)
        escrever_em_ficheiro_csv("./ficheiros/dist_manhattan_features_extraidas.csv", m_dist_man)
        escrever_em_ficheiro_csv("./ficheiros/dist_cosseno_features_extraidas.csv", m_dist_cos)

    
    #top 100 features
    if(os.path.exists("./ficheiros/dist_euclidiana_100_features.csv") == False and
        os.path.exists("./ficheiros/dist_manhattan_100_features.csv") == False and
         os.path.exists("./ficheiros/dist_cosseno_100_features.csv") == False):
        m_100_features = np.genfromtxt("./ficheiros/top100_features_normalizadas.csv", delimiter=",")
        
        m100_dist_euc = distancia_euclidiana(m_100_features)
        m100_dist_man = distancia_manhattan(m_100_features)
        m100_dist_cos = distancia_cosseno(m_100_features)
        
        escrever_em_ficheiro_csv("./ficheiros/dist_euclidiana_100_features.csv", m100_dist_euc)
        escrever_em_ficheiro_csv("./ficheiros/dist_manhattan_100_features.csv", m100_dist_man)
        escrever_em_ficheiro_csv("./ficheiros/dist_cosseno_100_features.csv", m100_dist_cos)
    #--------------------------------------------------------------------------------------------
        
        
        
    #Criar ranking
    if(CRIAR_RANKING_DIFERENCAS):
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
        np.savetxt("./ficheiros/rankings/ranking_features_extraidas_euclidiana.csv", m_ranking_100_euc[1:], 
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
    
    """    
    if (OUVIR_4_2):
        
        playing("MT0000202045.mp3", "./ficheiros/rankings/ranking_100_features_cosseno.csv")   
    """
        
    if(os.path.exists("./ficheiros/mat_similaridade_metadados.csv") == False):
        matriz_meta_dados = np.genfromtxt('./ficheiros/panda_dataset_taffc_metadata.csv', dtype="str", delimiter=",")
        matriz_similaridade_metadados = similaridade_metadados(matriz_meta_dados)
        np.savetxt("./ficheiros/mat_similaridade_metadados.csv", matriz_similaridade_metadados, 
                   delimiter=',',fmt="%i")
        
        
    if(CRIAR_RANKING_METADADOS):
        q1 = "MT0000202045.mp3"
        q2 = "MT0000379144.mp3"
        q3 = "MT0000414517.mp3"
        q4 = "MT0000956340.mp3"
        
        mat_metadados = np.genfromtxt('./ficheiros/mat_similaridade_metadados.csv',  delimiter=",")
        
        m_ranking_metadados = criar_ranking_metadados(q1, q2, q3, q4, mat_metadados)
        np.savetxt("./ficheiros/rankings/ranking_metadados.csv", m_ranking_metadados, 
                   delimiter=',', fmt="%s")
        
        
        
    #calcular precisao
    if(CALCULAR_PRECISAO):
        #matriz metadados
        m_ranking_metadados = np.genfromtxt("./ficheiros/rankings/ranking_metadados.csv", dtype="str",delimiter=",")
        
        #100 features
        
        #euclididana
        ranking_100_euc = np.genfromtxt("./ficheiros/rankings/ranking_100_features_euclidiana.csv", dtype="str"
                                        ,delimiter=",")
        precisao_100_euc = calcular_precisao(ranking_100_euc, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_100_features_euclidiana.csv", precisao_100_euc,
                   delimiter = ",", fmt="%s")
        
        #manhattan
        ranking_100_man = np.genfromtxt("./ficheiros/rankings/ranking_100_features_manhattan.csv", dtype="str"
                                        , delimiter=",")
        precisao_100_man = calcular_precisao(ranking_100_man, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_100_features_manhattan.csv", precisao_100_man, 
                   delimiter = ",", fmt="%s")
        
        #cossenco
        ranking_100_cos = np.genfromtxt("./ficheiros/rankings/ranking_100_features_cosseno.csv", dtype="str"
                                        , delimiter=",")
        precisao_100_cos = calcular_precisao(ranking_100_cos, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_100_features_cosseno.csv", precisao_100_cos, 
                   delimiter = ",", fmt="%s")
        
        
        #features extraidas
        
        #euclididana
        ranking_extr_euc = np.genfromtxt("./ficheiros/rankings/ranking_features_extraidas_euclidiana.csv", dtype="str"
                                        ,delimiter=",")
        precisao_extr_euc = calcular_precisao(ranking_extr_euc, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_features_extraidas_euclidiana.csv", precisao_extr_euc,
                   delimiter = ",", fmt="%s")
        
        #manhattan
        ranking_extr_man = np.genfromtxt("./ficheiros/rankings/ranking_features_extraidas_manhattan.csv", dtype="str"
                                        , delimiter=",")
        precisao_extr_man = calcular_precisao(ranking_extr_man, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_features_extraidas_manhattan.csv", precisao_extr_man, 
                   delimiter = ",", fmt="%s")
        
        #cossenco
        ranking_extr_cos = np.genfromtxt("./ficheiros/rankings/ranking_features_extraidas_cosseno.csv", dtype="str"
                                        , delimiter=",")
        precisao_extr_cos = calcular_precisao(ranking_extr_cos, m_ranking_metadados)
        np.savetxt("./ficheiros/precisoes/precisao_features_extraidas_cosseno.csv", precisao_extr_cos, 
                   delimiter = ",", fmt="%s")



if __name__ == "__main__":
    plt.close('all')
    main()
    
    
    
        
        
