import numpy as np
import scipy.signal as sp

def PV_TSM(x, alpha, N= 2048, hop=512, fs=44100):
    """
        Modifica la duracion de un archivo de audio a traves del metodo PV-TSM y la ventana de Hann sin modificar el
        pitch.

        Parameters
        ----------
        x : numpy array
            array del archivo que se quiere procesar

        alpha: float
            factor de compresion o retraso. Para comprimir alpha < 1, para expandir alpha > 1

        N: int>0
            ancho de la ventana

        hop: int>0
            salto de la ventana. Usualmente la mitad de N

        fs = int
            frecuencia de muestreo del sistema

        Returns
        -------
        y: numpy array
            array que contiene el audio comprimido o retrasado

        """
    L = len(x)  # Cantidad de muestras de 'x'

    # Comprimir:    alpha > 1
    # Expandir:     alpha < 1

    syn_hop = hop
    an_hop = int(syn_hop / alpha)  # H_a
    m_max = int((L - N) / an_hop)  # Valor límite de 'm' según el largo 'L' y el ancho de ventana 'N'
    delta_t = an_hop / fs

    f, t, X = sp.stft(x, nperseg=N, window='hann')
    X = X.T  # Traspongo la matriz para que tenga forma [m,k]
    f_coef = f * fs / N

    # Modulo y fase de X
    X_abs = np.abs(X)
    X_ang = np.angle(X)

    f_inst = np.zeros((len(t), len(f)))
    for k in range(len(f)):
        for m in range(len(t) - 1):
            f_inst[m, k] = f_coef[k] + (X_ang[m + 1, k] - X_ang[m, k] - f_coef[k] * delta_t) / (delta_t * np.pi * 2)

    X_ang_mod = np.zeros((len(t), len(f)))
    X_ang_mod[0, :] = X_ang[0, :]

    for m in range(len(t) - 1):
        for k in range(1, len(f)):
            X_ang_mod[m + 1, k] = X_ang_mod[m, k] + f_inst[m, k] * syn_hop / fs

    X_mod = (X_abs * np.exp(1j * X_ang_mod)).T

    x_mod = sp.istft(X_mod, window='hann')[1]
    # Defino el tamaño del array de salida y
    y = np.zeros(int(np.ceil(L * alpha)))

    for i in range(m_max):
        x_m = x_mod[i * an_hop:i * an_hop + N]
        
        if len(x_m) < N:
            x_m = np.hstack([x_m, np.zeros(N - len(x_m))])  # Rellena con ceros si es necesario
            
        x_m = x_m * np.hanning(N)
        
        if i * syn_hop + N <= len(y):
            y[i * syn_hop: i * syn_hop + N] += x_m
            
        else:
            resto = len(y) - i * syn_hop          #Corrige  la longitud de y en los últimos frames
            y[i * syn_hop: len(y)] += x_m[:resto]
            break

    return y

if __name__ == '__main__':
    
    from Graficadora_Audio import signal_graph
    from Carga_Audio import carga_wav

    lista = ['test_s1.wav', 'test_s2.wav']
    archivos = carga_wav(lista)
    audiodata, fsamp = archivos.get('test_s2.wav')
    fmod = PV_TSM(audiodata, 5, 2048, 512, fsamp)
    signal_graph(fmod, title='Señal modificada en tiempo')




