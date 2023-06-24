import numpy as np
import soundfile as sf

def lectura_wav(filename):
    '''
   Parameters
    ----------
    filename : archivo de audio

    Returns
    -------
    data : numpy array
        Array con información del audio.

    samplerate: int
        Frecuencia de muestreo del audio.
    '''

    data, samplerate = sf.read(filename)

    return data, samplerate


def carga_wav(files):
    """
    Almacena los datos de los archivos de audio .wav que se ingresen.

    Parameters
    ----------
    files : numpy array o lista
        Lista o array de numpy con los nombres de los archivos de audio.

    Returns
    -------
    audios : dict
        Diccionario con los nombres de los archivos como keys para almacenar los datos. Es una tupla:
        Primero contiene el array de información del archivo primero y despues la frecuencia
        de sampleo después (int).

    Ejemplo de como extraer la información del diccionario
    -------------------------------------------------------
    lista = ['audio.wav','audio2.wav','audio3.wav']

    archivos = carga_wav(lista)

    audiodata, fsamp = archivos.get('audio.wav')
    audiodata2, fsamp2 = archivos.get('audio2.wav')
    audiodata3, fsamp3 = archivos.get('audio3.wav')

    """

    files = np.array(files)
    audios = {}

    for i in range(files.size):
        data, samplerate = lectura_wav(files[i])
        audios[files[i]] = data, samplerate

    return audios

if __name__ == '__main__':

    from Graficadora_Audio import signal_graph

    lista = ['test_s1.wav','test_s2.wav']
    archivos = carga_wav(lista)
    audiodata, fsamp = archivos.get('test_s1.wav')
    signal_graph(audiodata/fsamp,fsamp)