import os
import speech_recognition as sr
from pydub import AudioSegment


def prepare_voice_file(path: str) -> str:
    """
    Converts the input audio file to WAV format if necessary and returns the path to the WAV file.
    """
    if os.path.splitext(path)[1] == '.wav':
        return path
    elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
        audio_file = AudioSegment.from_file(
            path, format=os.path.splitext(path)[1][1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(
            f'Unsupported audio format: {format(os.path.splitext(path)[1])}')


def transcribe_audio(audio_data, lang) -> str:
    """
    Transcribes audio data to text using Google's speech recognition API.
    """
    r = sr.Recognizer()
    text = r.recognize_google(audio_data, language=lang)
    return text


def write_transcription_to_file(text, output_file) -> None:
    """
    Writes the transcribed text to the output file.
    """
    with open(output_file, 'w') as f:
        f.write(text)


def clean_transcription(text, stopwords=None):
    """
    Cleans the transcribed text by removing specified stopwords.

    Parameters:
        text (str): The transcribed text.
        stopwords (list of str): A list of stopwords to be removed from the text.

    Returns:
        str: The cleaned text.
    """
    if stopwords is None:
        stopwords = ["the", "a", "an"]

    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def speech_to_text(input_path: str, lang: str) -> str:
    """
    Transcribes an audio file at the given path to text and returns the transcribed text as a string.
    """
    # Prepare the voice file
    wav_file = prepare_voice_file(input_path)

    # Load audio data from the prepared voice file
    with sr.AudioFile(wav_file) as source:
        audio_data = sr.Recognizer().record(source)

    # Transcribe the audio data
    text = transcribe_audio(audio_data, lang)

    # Clean the transcribed text
    cleaned_text = clean_transcription(text)

    # Return the cleaned transcribed text
    return cleaned_text


"""
if __name__ == '__main__':
    print('Please enter the path to an audio file (WAV, MP3, M4A, OGG, or FLAC):')
    input_path = input().strip()
    if not os.path.isfile(input_path):
        print('Error: File not found.')
        exit(1)
    else:
        print('Please enter the path to the output file:')
        output_path = input().strip()
        print('Please enter the language code (e.g. en-US):')
        language = input().strip()
        try:
            speech_to_text(input_path, lang)
        except Exception as e:
            print('Error:', e)
            exit(1)
"""
