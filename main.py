import record
import timing
import transcribe
import distance
import speech_recognition as sr


def process_audio_and_calculate_distances(recording_path_template, words_vector, lang="en-US"):
    """
    Records audio, detects speech timestamps, transcribes the audio to text, and calculates word distances.

    Parameters:
        recording_path_template (str): The template path to save the recorded audio.
        words_vector (list of str): A list of words to calculate distances from the transcribed text.
        lang (str): The language for speech transcription. Default is "en-US".

    Returns:
        tuple: Two dictionaries - one with words and their distances from the transcribed text,
               and another with words and their corresponding timestamps.
    """
    transcriptions = []
    latencies = {}

    # Step 1 to 3: Record, timestamp, and transcribe audio for each word in words_vector
    for idx, word in enumerate(words_vector):
        print(f"Processing word: {word}")

        recording_path = recording_path_template.format(idx)

        # Record audio
        record.record_audio(recording_path)

        # Detect speech timestamps
        timestamp = timing.detect_speech_timestamps(recording_path)
        timestamp = next((entry['start'] for entry in timestamp if 'start' in entry), None)
        print(f"Timestamps for recording {idx}:", timestamp)

        # Transcribe audio to text with error handling
        try:
            transcribed_text = transcribe.speech_to_text(input_path=recording_path, lang=lang).lower()
        except sr.UnknownValueError:
            transcribed_text = "None"
        print(f"Transcription for recording {idx}:", transcribed_text)

        # Store the transcription
        transcriptions.append(transcribed_text)

        # Ensure the word key is initialized as a dictionary in latencies
        if word not in latencies:
            latencies[word] = {}
        latencies[word][transcribed_text] = timestamp

    # Step 4: Calculate distances between transcriptions and the corresponding words
    distances = {}
    for idx, transcribed_text in enumerate(transcriptions):
        word = words_vector[idx]
        if transcribed_text == "None":
            distance_value = "None"
        else:
            distance_value = distance.calculate_word_distance(transcribed_text, word)
        # Ensure the word key is initialized as a dictionary in distances
        if word not in distances:
            distances[word] = {}
        distances[word][transcribed_text] = distance_value
        print(f"Distance between transcription '{transcribed_text}' and word '{word}': {distance_value}")

    return distances, latencies


# Example usage:
recording_path_template = "/Users/dennisklose/PycharmProjects/decodePSY/recording{}.wav"
words_vector = ["panda", "song", "festive", "carol"]
distances, latencies = process_audio_and_calculate_distances(recording_path_template, words_vector)
print(distances)
print(latencies)

#take only first word of the transcription and also make sure that in case word is too rare for glossary of distances, that the error is handled