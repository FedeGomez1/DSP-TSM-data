import numpy as np


def OLA(x, alpha, N=1024, hop=512, fs=44100):
    """
        Modifica la duracion de un archivo de audio a traves del metodo TSM OLA y la ventana de Hann sin modificar el
        pitch.

        Parameters
        ----------
        x : numpy array
            array del archivo que se quiere procesar

        alpha: 0<float<2
            factor de compresion o retraso. Para comprimir alpha < 1, para expandir alpha > 1

        N: int>0
            ancho de la ventana

        hop: int>0
            salto de la ventana. Usualmente la mitad de N

        fs = int
            frecuencia de muestreo del sistema

        Returns
        -------
        y : numpy array
            array que contiene el audio comprimido o retrasado

        """

    L = len(x)                  # Cantidad de muestras de 'x'

    an_hop = int(hop/alpha)  # H_a

        # Comprimir:    alpha < 1
        # Expandir:     alpha > 1

    m_max = int((L - N) / an_hop)   # Valor límite de 'm' segun el largo 'L' y el ancho de ventana 'N'

    x_m = np.zeros((m_max, N))

    for m in range(m_max):
        for r in range(N):
            x_m[m, r] = x[r + m * an_hop]

    # Aplico la ventana de Hann a x_m, obteniendo y_m
    window = np.hanning(N)
    y_m = x_m * window

    # Armo el array de salida 'y'
    y = np.zeros(int(L * alpha))

    for m in range(m_max):
        for r in range(N):
            if y.shape[0] % N != 0:
                y = np.hstack([y,np.zeros(N - y.shape[0] % N)]) # Agrego ceros en caso de que la longitud de la señal no sea divisible por la longitud de ventana
            y[r + m * hop] += y_m[m, r]  # Overlap y suma

    return y

if __name__ == '__main__':

    from Graficadora_Audio import signal_graph
    from Carga_Audio import carga_wav

    lista = ['test_s1.wav', 'test_s2.wav']
    archivos = carga_wav(lista)
    audiodata, fsamp = archivos.get('test_s2.wav')
    fmod = OLA(audiodata, 1.68745, 1024, 512,fsamp)
    signal_graph(fmod, title='Señal modificada en tiempo')

