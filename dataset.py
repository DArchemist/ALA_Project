import numpy as np

NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
mapa_notas = {nota: i for i, nota in enumerate(NOTAS)}

BEMOLES_A_SOSTENIDOS = {
    'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
    'Cb': 'B', 'Fb': 'E', 'E#': 'F', 'B#': 'C'
}

def nota_a_indice(nota):
    if nota in BEMOLES_A_SOSTENIDOS:
        nota = BEMOLES_A_SOSTENIDOS[nota]
    return mapa_notas.get(nota, -1)

INTERVALOS_POR_SUFIJO = {
    '':      [0, 4, 7],          # Tríada Mayor (C)
    'm':     [0, 3, 7],          # Tríada Menor (Cm)
    '7':     [0, 4, 7, 10],      # Séptima de Dominante (C7)
    'maj7':  [0, 4, 7, 11],      # Séptima Mayor (Cmaj7)
    'm7':    [0, 3, 7, 10],      # Séptima Menor (Cm7)
    'dim':   [0, 3, 6],          # Tríada Disminuida (Cdim)
    'aug':   [0, 4, 8],          # Tríada Aumentada (Caug)
}

def parsear_acorde(acorde_str):

    if len(acorde_str) >= 2 and acorde_str[1] == '#':
        raiz = acorde_str[:2]
        sufijo = acorde_str[2:]
    else:
        raiz = acorde_str[0]
        sufijo = acorde_str[1:]
    
    return raiz, sufijo

def acorde_a_vector(acorde_str):
    """
    Genera un vector de R^12 con unos en las notas del acorde.
    Usa intervalos modulares (módulo 12) para construir las notas.
    
    Ejemplo:
        acorde_a_vector('C') -> [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        (Notas: C, E, G -> índices 0, 4, 7)
    """
    raiz, sufijo = parsear_acorde(acorde_str)
    
    idx_raiz = nota_a_indice(raiz)
    if idx_raiz == -1:
        raise ValueError(f"Raíz '{raiz}' no reconocida en {acorde_str}")
    
    if sufijo not in INTERVALOS_POR_SUFIJO:
        print(f"Advertencia: Sufijo '{sufijo}' no reconocido para '{acorde_str}'. Usando mayor.")
        intervalos = INTERVALOS_POR_SUFIJO['']
    else:
        intervalos = INTERVALOS_POR_SUFIJO[sufijo]
    
    indices_notas = [(idx_raiz + intervalo) % 12 for intervalo in intervalos]
    
    vector = np.zeros(12, dtype=float)
    for idx in indices_notas:
        vector[idx] = 1.0
    
    return vector

def vectorizar_y_normalizar(acorde_str):
    v = acorde_a_vector(acorde_str)
    norma = np.linalg.norm(v) # Aquí se usa la norma euclidiana, p = 2
    if norma > 0:
        v = v / norma
    return v


def construir_dataset():
    raices = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    sufijos = ['', 'm', '7', 'maj7', 'm7', 'dim', 'aug']
    
    acordes_str = []
    for r in raices:
        for s in sufijos:
            acordes_str.append(r + s)
    
    X = []
    nombres = []
    for acorde in acordes_str:
        v = vectorizar_y_normalizar(acorde)
        X.append(v)
        nombres.append(acorde)
    
    X = np.array(X)
    return X, nombres