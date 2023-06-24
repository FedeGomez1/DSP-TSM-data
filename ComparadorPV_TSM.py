import pytsmod
import matplotlib.pyplot as plt
import numpy as np
from PV_TSM import PV_TSM

def ComparadorPV_TSM(x, alpha, N=2048, hop=512, fs=44100):
    """
            Grafica el procesamiento de un archivo de audio a traves de diferentes metodos de PV-TSM para ser
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

                fs = int
                    frecuencia de muestreo del sistema

                Returns
                -------
                y1 : numpy array
                    array que contiene el audio comprimido o retrasado de la funcion de la libreria pytsmod

                y2 : numpy array
                    array que contiene el audio comprimido o retrasado de la funcion creada por nosotros
                """
    y1 = pytsmod.phase_vocoder(x, alpha, win_type='hann', win_size=N, syn_hop_size=hop)
    y2 = PV_TSM(x, alpha, N, hop, fs)

    # Obtener los valores de tiempo para el eje x
    t1 = np.linspace(0, len(y1) / fs, len(y1))
    t2 = np.linspace(0, len(y2) / fs, len(y2))

    # Graficar las funciones individualmente
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))

    # Graficar y1
    axs[0].plot(t1, y1, label='pytsmod.phase_vocoder()')
    axs[0].set_xlabel('Tiempo (s)')
    axs[0].set_ylabel('Amplitud')
    axs[0].set_title('Función de PyTSM')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar y2
    axs[1].plot(t2, y2, label='own PV-TSM')
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
    plt.plot(t1, y1, label='pytsmod.phase_vocoder')
    plt.plot(t2, y2, label='own PV-TSM')
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
    pytsm, propia = ComparadorPV_TSM(audiodata, 0.7, 2048, 512, fsamp)

    #Creacion de archivo de audio
    sf.write('test_s2(viapytsm).wav', pytsm, fsamp)
    sf.write('test_s2(viaowncode).wav', propia, fsamp)