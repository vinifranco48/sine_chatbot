from groq import Groq, APIError
import os
import sys
from src.config.settings import Settings

settings = Settings()


class SpeechToTextService:
    def __init__(self):
        self.api_key = settings.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        self.client = Groq(api_key=self.api_key)

    def transcribe_audio(self, file_path):
        """
        Transcribe audio file using Groq's Whisper model.
        
        Args:
            file_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text as string
        """
        try:
            with open(file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo"
                )
            return transcription.text
        except Exception as e:
            raise ValueError(f"Error transcribing audio: {str(e)}")