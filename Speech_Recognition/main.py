from dotenv import load_dotenv
import os

import requests

#upload

#transcribe

#poll

#save transcript



# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

load_dotenv('.env.local')

api_key = os.getenv('API_KEY')

aai.settings.api_key = api_key
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("./output_test.wav")

#to get it from audio file
# from pydub import AudioSegment
# AudioSegment.from_mp3("./Voice_Sample.mp3").export("./Voice_Sample.wav", format="wav")
# transcript = transcriber.transcribe("./Voice_Sample.wav")

print(transcript.text)

with open('Output_Transcript.txt', 'w') as f:
    f.write(transcript.text)