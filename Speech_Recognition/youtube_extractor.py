import youtube_dl
from pytube import YouTube
import assemblyai as aai
import os

def download_video(url):
    youtube = YouTube(url)
    video = youtube.streams.first()
    audio_stream = youtube.streams.filter(only_audio=True).first()
    filename = audio_stream.download()
    print("Audio URL:", audio_stream.url)
    return filename
    # filename = video.download()
    # print("Audio URL:", video.url)
    # return filename

if __name__ == "__main__":
    video_filename = download_video("https://www.youtube.com/watch?v=SPiwFAnDILU")
    print(video_filename)
    
    


aai.settings.api_key = os.getenv('API_KEY')

# audio_url = audio_stream.url
FILE_URL = '../Gravity_Pro.mp3'

config = aai.TranscriptionConfig(sentiment_analysis=True)

transcript = aai.Transcriber().transcribe(audio_url, config)

for sentiment_result in transcript.sentiment_analysis:
    print(sentiment_result.text)
    print(sentiment_result.sentiment)  # POSITIVE, NEUTRAL, or NEGATIVE
    print(sentiment_result.confidence)
    print(f"Timestamp: {sentiment_result.start} - {sentiment_result.end}")
