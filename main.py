from dataset import construir_dataset
import numpy as np

X, nombres = construir_dataset()

# Cálculo del rango de la matriz X
rango = np.linalg.matrix_rank(X)
print(f'\n El rango de la matriz X es: {rango}')

# Centrado de los datos
media_columnas = np.mean(X, axis=0)
X_cent = X - media_columnas

# Cálculo de la matriz de covarianza
m = X_cent.shape[0]
C = (1 / (m - 1)) * (X_cent.T @ X_cent)

# Cálculo del número de condición de la matriz de covarianza
condicion_C = np.linalg.cond(C)
print(f'\n La condición de la matriz de covarianza C es: {condicion_C:.2f}')

# Aplicación de SVD a la matriz centrada
U, S, Vt = np.linalg.svd(X_cent, full_matrices=False)

# Cálculo de la varianza explicada acumulada
var_exp = (S**2) / np.sum(S**2)
var_exp_acum = np.cumsum(var_exp)

print("\n Varianza explicada ACUMULADA:")
print(f"   PC1+PC2 (2D): {var_exp_acum[1]*100:.2f}%")
print(f"   PC1+PC2+PC3 (3D): {var_exp_acum[2]*100:.2f}%")
print(f"   PC1+PC2+PC3+PC4 (4D): {var_exp_acum[3]*100:.2f}%")
print(f"   PC1 a PC5 (5D): {var_exp_acum[4]*100:.2f}%")
print(f"   PC1 a PC6 (6D): {var_exp_acum[5]*100:.2f}%")

# Análisis de progresiones armónicas
progresiones = {
    "Ciclo de quintas (I - IV - V - I)": ['C', 'F', 'G', 'C'],
    "Misma raíz (C, Cm, C7, Cmaj7)": ['C', 'Cm', 'C7', 'Cmaj7'],
    "Pop/Rock (I - vi - ii - V)": ['Cmaj7', 'Am7', 'Dm7', 'G7'],
    "Sustitución de tritono (G7 -> C#7 -> C)": ['G7', 'C#7', 'Cmaj7'],
    "Mediantes cromáticas (C -> G# -> Fm -> G)": ['C', 'G#', 'Fm', 'G']
}

# Cálculo y comparación de distancias (norma p = 2)
def obtener_vector(acorde):
    idx = nombres.index(acorde)
    return X[idx, :]

def distancia_original(acorde1, acorde2):
    v1 = obtener_vector(acorde1)
    v2 = obtener_vector(acorde2)
    return np.linalg.norm(v1 - v2)

def distancia_reducida(acorde1, acorde2, k):
    idx1 = nombres.index(acorde1)
    idx2 = nombres.index(acorde2)
    
    v1_cent = X_cent[idx1, :]
    v2_cent = X_cent[idx2, :]
    
    coords1 = v1_cent @ Vt[:k, :].T
    coords2 = v2_cent @ Vt[:k, :].T
    
    return np.linalg.norm(coords1 - coords2)

def error_distancia(acorde1, acorde2, k):
    d_orig = distancia_original(acorde1, acorde2)
    d_red = distancia_reducida(acorde1, acorde2, k)
    if d_orig > 1e-10:
        return abs(d_red - d_orig) / d_orig * 100
    return 0

dimensiones = [2, 3, 4, 5, 6]
resultados = []

for nombre_prog, acordes in progresiones.items():
    print(f"\n  {nombre_prog}")
    print(f"   Acordes: {' -> '.join(acordes)}")
    
    pares = []
    for i in range(len(acordes)):
        for j in range(i+1, len(acordes)):
            pares.append((acordes[i], acordes[j]))
    
    dist_orig = [distancia_original(a, b) for a, b in pares]
    
    errores_por_dim = {}
    for k in dimensiones:
        errores = [error_distancia(a, b, k) for a, b in pares]
        errores_por_dim[f'{k}D'] = np.mean(errores)
    
    # Guardar resultados
    resultados.append({
        'Progresión': nombre_prog,
        'N° pares': len(pares),
        'Error 2D (%)': errores_por_dim['2D'],
        'Error 3D (%)': errores_por_dim['3D'],
        'Error 4D (%)': errores_por_dim['4D'],
        'Error 5D (%)': errores_por_dim['5D'],
        'Error 6D (%)': errores_por_dim['6D']
    })
    

print("\n TABLA RESUMEN DE ERRORES POR PROGRESIÓN")


# Crear tabla con formato
print(f"{'Progresión':<35} {'2D':>8} {'3D':>8} {'4D':>8} {'5D':>8} {'6D':>8}")
print("-" * 75)
for r in resultados:
    print(f"{r['Progresión'][:34]:<35} "
          f"{r['Error 2D (%)']:>7.2f}% "
          f"{r['Error 3D (%)']:>7.2f}% "
          f"{r['Error 4D (%)']:>7.2f}% "
          f"{r['Error 5D (%)']:>7.2f}% "
          f"{r['Error 6D (%)']:>7.2f}%"
    )

# Errores globales

error_global_2d = np.mean([r['Error 2D (%)'] for r in resultados])
error_global_3d = np.mean([r['Error 3D (%)'] for r in resultados])
error_global_4d = np.mean([r['Error 4D (%)'] for r in resultados])
error_global_5d = np.mean([r['Error 5D (%)'] for r in resultados])
error_global_6d = np.mean([r['Error 6D (%)'] for r in resultados])


print("\n ERROR GLOBAL PROMEDIO (TODAS LAS PROGRESIONES)")

print(f"   2D: {error_global_2d:.2f}%")
print(f"   3D: {error_global_3d:.2f}%")
print(f"   4D: {error_global_4d:.2f}%")
print(f"   5D: {error_global_5d:.2f}%")
print(f"   6D: {error_global_6d:.2f}%")




# Funciones de síntesis de audio y reconstrucción

def vector_a_audio(vector, duracion=1.2, sample_rate=44100, amplitud=0.3):
    frecuencias = [
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
        369.99, 392.00, 415.30, 440.00, 466.16, 493.88
    ]
    
    t = np.linspace(0, duracion, int(sample_rate * duracion), endpoint=False)
    audio = np.zeros_like(t)
    
    for i, valor in enumerate(vector):
        if valor > 0.05:  # Umbral para evitar ruido de reconstrucción
            freq = frecuencias[i]
            # Usar el valor como amplitud (1.0 en original, fracciones en reconstruido)
            audio += valor * np.sin(2 * np.pi * freq * t)
    
    # Normalizar para evitar saturación
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def reconstruir_acorde(nombre_acorde, k):
    idx = nombres.index(nombre_acorde)
    v_cent = X_cent[idx, :]
    coords_k = v_cent @ Vt[:k, :].T
    v_rec_cent = coords_k @ Vt[:k, :]
    v_rec = v_rec_cent + media_columnas
    v_rec = np.maximum(v_rec, 0)
    return v_rec

def generar_audio_progresion(acordes, k=None, duracion_acorde=1.2, silencio=0.3):
    sample_rate = 44100
    audios = []
    
    for acorde in acordes:
        if k is None:
            idx = nombres.index(acorde)
            vector = X[idx, :]
        else:
            vector = reconstruir_acorde(acorde, k)
        
        audio = vector_a_audio(vector, duracion=duracion_acorde, sample_rate=sample_rate)
        audios.append(audio)
        
        silence = np.zeros(int(silencio * sample_rate))
        audios.append(silence)
    
    audio_completo = np.concatenate(audios[:-1]) if audios else np.array([])
    
    return audio_completo

try:
    import soundfile as sf
    TENER_SOUNDFILE = True
except ImportError:
    TENER_SOUNDFILE = False
    exit()

dimensiones_audio = [
    ("original", None),
    ("2D", 2),
    ("3D", 3),
    ("4D", 4),
    ("5D", 5),
    ("6D", 6)
]

# Crear carpeta para los audios (opcional)
import os
carpeta_audios = "audios_progresiones"
if not os.path.exists(carpeta_audios):
    os.makedirs(carpeta_audios)

total_archivos = 0

for idx_prog, (nombre_prog, acordes) in enumerate(progresiones.items(), 1):
    for dim_nombre, k in dimensiones_audio:
        nombre_limpio = nombre_limpio = nombre_prog.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('/', '_')
        nombre_archivo = f"{carpeta_audios}/prog_{idx_prog:02d}_{nombre_limpio}_{dim_nombre}.wav"

        audio = generar_audio_progresion(acordes, k=k)
        
        sf.write(nombre_archivo, audio, 44100)
        total_archivos += 1