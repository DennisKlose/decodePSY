import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import transcribe

freq = 16000
duration=5
print("speak")
recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
sd.wait()
# before saving make all integers in array 32 bit
y = (np.iinfo(np.int32).max * (recording/np.abs(recording).max())).astype(np.int32)
write("recording0.wav", freq, y)
print("successfully recorded")

transcribe.speech_to_text("/Users/dennisklose/PycharmProjects/decodePSY/recording0.wav","/Users/dennisklose/PycharmProjects/decodePSY/test.txt","en-US")