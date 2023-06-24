
import numpy as np
import scipy.signal as sp


def HPS(x, ker, fs=44100):
    """
        Modifica la duracion de un archivo de audio a traves del metodo TSM HPS y la ventana de Hann sin modificar
        el pitch separando la componente armonica de la percursiva para procesarlas de manera separada y que el
        resultado sea mejor.

        Parameters
        ----------
        x : numpy array
            array del archivo que se quiere procesar

        ker: 0<int<100
            Tamaño del kernel para el filtrado de media móvil (Tiene que ser impar). A mayor kernel, mayor separacion

        fs = int
            frecuencia de muestreo del sistema

        Returns
        -------
        x_h : numpy array
            array que contiene la parte armonica (harmonic) del audio
        x_p : numpy array
            array que contiene la parte percusiva (percusive) del audio

        """
    f, t, X = sp.stft(x, fs)  # Obtiene los arrays de frecuencia y tiempo y el de la STFT del audio
    Sxx = np.abs(X)   #  Calcula el espectrograma del audio
    kernel = ker

    Sxx_h = np.zeros(shape = (len(f),len(t)))     # Defino los espectrogramas a filtrar
    Sxx_p = np.zeros(shape = (len(f),len(t)))

    for m in range(len(t)):   # Aplico filtro de media móvil a lo largo del array de frecuencia para cada instante
        Sxx_p[:,m] = sp.medfilt(Sxx[:,m], kernel_size = kernel)

    for k in range(len(f)):   # Aplico filtro de media móvil a lo largo del tiempo para cada frecuencia
        Sxx_h[k,:] = sp.medfilt(Sxx[k,:], kernel_size = kernel)


    M_h= np.zeros(shape = (len(f),len(t)))   # Defino las máscaras binarias como matrices
    M_p= np.zeros(shape = (len(f),len(t)))

    for m in range(len(t)):
        for k in range(len(f)):
            if Sxx_p[k,m] >= Sxx_h[k,m]:     # Establezco la condición de las máscaras binarias
                M_p[k,m] = 1
            else:
                M_h[k,m] = 1

    X_h = X*M_h    # Multiplico la STFT por cada máscara, obteniendo las componentes armónica y percusiva del audio
    X_p = X*M_p

    x_h = sp.istft(X_h)[1] # Antitransformo los arrays
    x_p = sp.istft(X_p)[1]

    return x_h, x_p

if __name__ == '__main__':

    from Graficadora_Audio import signal_graph
    from Carga_Audio import carga_wav
    from OLA_TSM import OLA
    from PV_TSM import PV_TSM

    lista = ['test_s1.wav', 'test_s2.wav']
    archivos = carga_wav(lista)
    audiodata, fsamp = archivos.get('test_s2.wav')
    armonica, percusiva = HPS(audiodata, 31, fsamp)

    fmod = OLA(percusiva, 0.5, 1024, 512, fsamp) + PV_TSM(armonica, 0.5, 1024, 512, fsamp)

    signal_graph(fmod,title='Señal modificada en tiempo')