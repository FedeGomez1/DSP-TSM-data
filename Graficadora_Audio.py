import matplotlib.pyplot as plt
import numpy as np

def signal_graph(data, fs=44100, title=None):
    '''
    Grafica una señal en el tiempo.

    Parametros
    ----------
    data: array
        Datos de la amplitud de la señal.
    fs: int
        Frecuencia de muestreo de la señal en Hz. Por defecto el valor es 44100 Hz.
    '''

    x = np.linspace(0, len(data)/fs, len(data))
    y = data
    plt.plot(x,y,'-')
    plt.xlabel("Tiempo[s]")
    plt.ylabel("Amplitud normalizada")
    plt.title(title)    
    plt.show()