from HPS import HPS
import pytsmod
import matplotlib.pyplot as plt
import numpy as np
from OLA_TSM import OLA
from PV_TSM import PV_TSM

def ComparadorHPS(x, alpha, N=1024, hop=512, ker=31, fs=44100):
    """
            Grafica el procesamiento de un archivo de audio a traves de diferentes metodos de TSM HPS para ser
            comparados visualmente.

            Parameters
            ----------
            x : numpy array
                array del archivo que se quiere procesar

            alpha: 0<float<2
                factor de compresion o retraso. Para comprimir alpha < 1, para expandir alpha > 1

            N: int>0
                ancho de la ventana. Usualmente N = 1024

            hop: int>0
                salto de la ventana. Usualmente la mitad de N

            ker: 0<int<100
                Tamaño del kernel para el filtrado de media móvil (Tiene que ser impar). A mayor kernel, mayor separacion

            fs = int
                frecuencia de muestreo del sistema

            Returns
            -------
            y1 : numpy array
                array que contiene el audio comprimido o retrasado de la funcion de la libreria pytsmod

            y2 : numpy array
                array que contiene el audio comprimido o retrasado de la funcion creada por nosotros

            """

    y1 = pytsmod.hptsm(x, alpha) #función de pytsmod

    armonica, percusiva = HPS(x, ker, fs) #division de función propia

    y2 = OLA(percusiva, alpha, N, hop,fs) + PV_TSM(armonica, alpha, N, hop, fs) #armado de función propia

    # Obtener los valores de tiempo para el eje x
    t1 = np.linspace(0, len(y1)/fs, len(y1))
    t2 = np.linspace(0, len(y2) / fs, len(y2))

    # Graficar las funciones individualmente
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))

    # Graficar y1
    axs[0].plot(t1, y1, label='pytsmod.hptsm')
    axs[0].set_xlabel('Tiempo (s)')
    axs[0].set_ylabel('Amplitud')
    axs[0].set_title('Función de PyTSM')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar y2
    axs[1].plot(t2, y2, label='own HPS')
    axs[1].set_xlabel('Tiempo (s)')
    axs[1].set_ylabel('Amplitud')
    axs[1].set_title('Función propia')
    axs[1].legend()
    axs[1].grid(True)

    # Mostrar los gráficos individuales
    plt.tight_layout()
    plt.show()

    # Graficar todas las funciones juntas
    plt.figure(figsize=(10, 6))
    plt.plot(t1, y1, label='pytsmod.hptsm')
    plt.plot(t2, y2, label='own HPS')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Gráfico con todas las funciones')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y1, y2

if __name__ == '__main__':

    from Carga_Audio import carga_wav
    import soundfile as sf

    lista = ['test_s1.wav', 'test_s2.wav']
    archivos = carga_wav(lista)
    audiodata, fsamp = archivos.get('test_s2.wav')
    pytsm, propia = ComparadorHPS(audiodata, 0.5, 1024, 512, 31, fsamp)

    # Creacion de archivo de audio
    sf.write('test_s2(viapytsm).wav', pytsm, fsamp)
    sf.write('test_s2(viaowncode).wav', propia, fsamp)