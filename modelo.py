import numpy as np
import librosa
from python_speech_features import mfcc
import os

def segmentAudio(data, sr, seg_duration=0.05, hop_duration=0.025):

    seg_length = int(seg_duration * sr)
    hop_length = int(hop_duration * sr)
    segments = []
    for i in range(0, len(data) - seg_length + 1, hop_length):
        segment = data[i : i + seg_length]
        
        energy = np.sum(segment**2) / len(segment)
        if energy > 0.00001:
            segments.append(segment)
    return segments

def zero_crossing_rate(signal):
    return np.mean(np.abs(np.diff(np.sign(signal))))

def extractFeatures(segments, sr):
    features = []
    for segment in segments:
        energy = np.sum(segment**2) / len(segment)
        zcr_val = zero_crossing_rate(segment)
        mfcc_feat = mfcc(segment, samplerate=sr, numcep=12, nfft=1024)
        mfcc_mean = np.mean(mfcc_feat, axis=0)
        feature_vector = [energy, zcr_val] + mfcc_mean.tolist()
        features.append(feature_vector)
    return np.array(features)

#analisis de señal, extraccion de caracteristicas, 
def Dataset(archivos, seg_duration=0.05, hop_duration=0.025):

    all_features = []
    for archivo in archivos:

        data, sr = librosa.load(archivo)
        print(f"{archivo}, duración = {len(data) / sr:.2f}s, sr = {sr}Hz")

        segments = segmentAudio(data, sr, seg_duration, hop_duration)

        features = extractFeatures(segments, sr)
        all_features.append(features)

    if len(all_features) > 0:
        all_features = np.vstack(all_features)
    else:
        all_features = np.array([])
    return all_features

#archivos="direccion dataset"

from hmmlearn import hmm

def entrenar_modelos_hmm(archivos_dict, seg_duration=0.05, hop_duration=0.025):
    modelos = {}
    for fonema, lista_archivos in archivos_dict.items():
        print(f"Entrenando HMM para fonema: {fonema}")
        features = Dataset(lista_archivos, seg_duration, hop_duration)

        if len(features) > 0:
            modelo = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=200)
            modelo.fit(features)
            modelos[fonema] = modelo
    return modelos

def clasificar_audio(modelos, archivo, seg_duration=0.05, hop_duration=0.025):
    data, sr = librosa.load(archivo)
    segmentos = segmentAudio(data, sr, seg_duration, hop_duration)
    features = extractFeatures(segmentos, sr)

    scores = {}
    for fonema, modelo in modelos.items():
        try:
            score = modelo.score(features)  
            scores[fonema] = score
        except:
            scores[fonema] = float('-inf')  
    predicho = max(scores, key=scores.get)
    return predicho, scores

def evaluar_modelos(modelos, test_dict):
    total = 0
    aciertos = 0
    for clase_real, archivos in test_dict.items():
        for archivo in archivos:
            clase_predicha, _ = clasificar_audio(modelos, archivo)
            total += 1
            if clase_predicha == clase_real:
                aciertos += 1
            print(f"Real: {clase_real}, Predicha: {clase_predicha}")
    accuracy = aciertos / total if total > 0 else 0
    print(f"\nAccuracy total: {accuracy*100:.2f}%")

from pathlib import Path

def cargar_dataset(ruta_dataset, extensiones_validas=(".wav")):
    dataset_path = Path(ruta_dataset)
    archivos_por_clase = {}

    for carpeta_clase in dataset_path.iterdir():
        if carpeta_clase.is_dir():
            archivos = list(carpeta_clase.glob("*"))
            archivos_filtrados = [str(a) for a in archivos if a.suffix in extensiones_validas]
            archivos_por_clase[carpeta_clase.name] = archivos_filtrados
    return archivos_por_clase

from sklearn.model_selection import train_test_split

def dividir_dataset(dataset_dict, test_size=0.2, random_state=42):
    train_dict = {}
    test_dict = {}
    for clase, archivos in dataset_dict.items():
        train_files, test_files = train_test_split(archivos, test_size=test_size, random_state=random_state)
        train_dict[clase] = train_files
        test_dict[clase] = test_files
    return train_dict, test_dict

dataset_dict = cargar_dataset("DataSet")
train_dict, test_dict = dividir_dataset(dataset_dict)
modelos = entrenar_modelos_hmm(dataset_dict)
evaluar_modelos(modelos, test_dict)

pred, scores = clasificar_audio(modelos, 'prueba.wav')
print("Palabra predicha:", pred)
print("Scores:", scores)
