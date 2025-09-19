import os
import io
from pytubefix import YouTube

from engine.processors.audio_processor import AudioTranscriber

class YouTubeProcessor:
    def __init__(self):
        pass

    def load_youtube_video(self, url):
        try:
            # Initialize YouTube object
            yt = YouTube(url)
            video_buffer = io.BytesIO()
            audio_buffer = io.BytesIO()

            print(f"Fetching: {yt.title}")

            # video_stream = yt.streams.filter(
            #     progressive=True, file_extension="mp4"
            # ).order_by("resolution").desc().first()

            # if video_stream:
            #     print(f"Loading video in {video_stream.resolution} into memory...")
            #     video_buffer = io.BytesIO()
            #     video_stream.stream_to_buffer(video_buffer)
            #     video_buffer.seek(0)
            #     print("Video loaded into memory ✅")
            # else:
            #     print("No suitable video stream found")

            audio_stream = yt.streams.filter(
                only_audio=True, file_extension="mp4"
            ).first()

            if audio_stream:
                print("Loading audio into memory...")
                audio_buffer = io.BytesIO()
                audio_stream.stream_to_buffer(audio_buffer)
                audio_buffer.seek(0)
                print("Audio loaded into memory ✅")
            else:
                print("No suitable audio stream found")

            return {"video": video_buffer, "audio": audio_buffer}
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Suggestions:")
            print("- Ensure the YouTube URL is valid and the video is publicly accessible.")
            print("- Update pytubefix: pip install --upgrade pytubefix")
            print("- Try a different video to rule out restrictions.")
            return {"video": None, "audio": None}

    def process_audio(self, url):
        media = self.load_youtube_video(url)
        audio_buffer = media['audio']
        transcriber = AudioTranscriber()
        transcript = transcriber.transcribe(audio_buffer)
        if transcript:
            print("Audio processed successfully.")
            return transcript
        else:
            print("Audio processing failed.")
            return None

if __name__ == "__main__":
    yt_url = "https://www.youtube.com/watch?v=fqvRz5u4QJE&list=RDfqvRz5u4QJE&start_radio=1"  # Example URL
    processor = YouTubeProcessor()
    transcript = processor.process_audio(yt_url)
    if transcript:
        print("Transcript:")
        print(transcript)
    else:
        print("No transcript available.")