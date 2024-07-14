import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(filename, freq=16000, duration=5):
    """
    Records audio and saves it to a WAV file.
    """

    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()
    # Before saving, convert all integers in the array to 32-bit
    y = (np.iinfo(np.int32).max * (recording / np.abs(recording).max())).astype(np.int32)
    write(filename, freq, y)

