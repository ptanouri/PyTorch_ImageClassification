# audio file formats
#.mp3 (compresses data, can lose information)
#.flac (cmpressed, loss less, but can construct original)
#.wav (uncompressed, best quality but large files)

import wave
import matplotlib.pyplot as plt
import numpy as np 

#audio signal parameters
# - number of channels
# - sample width
# - frame rate/sample rare 44,100 Hz
# - number of frames 
# - values of frame

# READ A WAV FILE

obj = wave.open('Voice_Sample.wav', 'rb')


channels = obj.getnchannels()
sample_width = obj.getsampwidth()
sample_frequncy = obj.getframerate()
n_samples = obj.getnframes()

print("number of channels", channels)
print("sample width", sample_width)
print("frame rate", sample_frequncy)
print("number of frames", n_samples)
print("parameters", obj.getparams())

t_audio = n_samples / sample_frequncy

print("time of audio", t_audio)

#read frames and print them

signal_wave = obj.readframes(-1)

obj.close()


#setting new audio file

obj_new = wave.open('Voice_Sample_New.wav', 'wb')
obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(48000.0)
obj_new.writeframes(signal_wave)
obj_new.close()


#Let's plot the audio signal

signal_array = np.frombuffer(signal_wave, dtype=np.int16)
times = np.linspace(0, t_audio, num=len(signal_array))

plt.figure(figsize=(10, 5))
plt.plot(times, signal_array, label='audio signal')
plt.title("Audio Signal Plot")
plt.xlabel("Time")
plt.ylabel("Signal Wave")
plt.xlim(0, t_audio)
plt.show()
