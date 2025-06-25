import tkinter as tk
from PIL import Image, ImageTk
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from python_speech_features import mfcc
from hmmlearn import hmm
import pickle

# === ARCHIVOS DE IMAGEN ===
ruta_pista = "pista.jpg"
ruta_carro = "carro.png"

# === POSICIONES ===
car_position_x = [60, 180, 300]  # izquierda, centro, derecha 
car_position_y = 200
current_position = 1  # carril central

# === GRABAR AUDIO ===
def grabar_audio(nombre_archivo='prueba.wav', duracion=2, sr=22050):
    print("🎙️ Grabando audio...")
    audio = sd.rec(int(duracion * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    write(nombre_archivo, sr, audio)
    print(f"✔️ Audio guardado como '{nombre_archivo}'")

# === FUNCIONES HMM ===
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
        mfcc_feat = mfcc(segment, samplerate=sr, numcep=12, nfft=1024)
        delta_feat = delta(mfcc_feat, 2)
        delta_delta_feat = delta(delta_feat, 2)
        mfcc_combined = np.hstack([
            np.mean(mfcc_feat, axis=0),
            np.mean(delta_feat, axis=0),
            np.mean(delta_delta_feat, axis=0)
        ])

        feature_vector = [energy, zcr_val] + mfcc_combined.tolist()
        features.append(feature_vector)
    return np.array(features)


def clasificar_audio(modelos, archivo):
    data, sr = librosa.load(archivo)
    segmentos = segmentAudio(data, sr)
    features = extractFeatures(segmentos, sr)

    scores = {}
    for fonema, modelo in modelos.items():
        try:
            score = modelo.score(features)
            scores[fonema] = score
        except:
            scores[fonema] = float('-inf')
    predicho = max(scores, key=scores.get)
    return predicho

with open("modelos_entrenados.pkl", "rb") as f:
    modelos = pickle.load(f)

ventana = tk.Tk()
ventana.title("Simulación de carriles")
ventana.geometry("400x600")

fondo_img = Image.open(ruta_pista).resize((400, 600))
fondo_tk = ImageTk.PhotoImage(fondo_img)

carro_img = Image.open(ruta_carro).resize((60, 120))
carro_tk = ImageTk.PhotoImage(carro_img)

canvas = tk.Canvas(ventana, width=400, height=600)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=fondo_tk)
carro_id = canvas.create_image(car_position_x[current_position], car_position_y, anchor=tk.NW, image=carro_tk)

def actualizar_carro():
    global current_position
    grabar_audio('prueba.wav')
    clase = clasificar_audio(modelos, 'prueba.wav')
    print("🔍 Comando reconocido:", clase)

    if clase == 'izquierda':
        current_position = 0
    elif clase == 'centro':
        current_position = 1
    elif clase == 'derecha':
        current_position = 2
    else:
        print("❌ Comando no reconocido.")

    canvas.coords(carro_id, car_position_x[current_position], car_position_y)

boton = tk.Button(ventana, text="🎙️ Grabar comando", font=("Arial", 14), command=actualizar_carro)
boton.pack(pady=10)

ventana.mainloop()