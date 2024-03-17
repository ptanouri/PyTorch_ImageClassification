from pydub import AudioSegment

audio = AudioSegment.from_file("Voice_Sample.wav", format="wav")

#increase the volume by 10 dB

louder_audio = audio + 20

louder_audio = louder_audio.fade_in(2000)

louder_audio.export("Voice_Sample.mp3", format="mp3")

audio2 = AudioSegment.from_file("Voice_Sample.mp3", format="mp3")

print("done")