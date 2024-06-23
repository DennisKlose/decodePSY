import streamlit as st
import record
import timing
import transcribe
import distance
import speech_recognition as sr
import time
import random
import pandas as pd

def create_words_vector(file_path, num_words=5):
    # Read all lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Remove newline characters and filter out empty lines
    words_list = [line.strip() for line in lines if line.strip()]

    # Ensure we have enough words to choose from
    num_lines = len(words_list)
    if num_words > num_lines:
        num_words = num_lines

    # Select a random sample of num_words from the list
    selected_words = random.sample(words_list, num_words)

    # Return the selected words as a vector
    return selected_words


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
    latencies = {}

    # Streamlit placeholders for displaying content
    word_placeholder = st.empty()
    transcription_placeholder = st.empty()

    # Step 1 to 3: Record, timestamp, and transcribe audio for each word in words_vector
    for idx, word in enumerate(words_vector):
        # Display the current word
        word_placeholder.markdown(f"<h1 style='text-align: center;'>{word}</h1>", unsafe_allow_html=True)

        recording_path = recording_path_template.format(idx)

        # Record audio
        record.record_audio(recording_path)

        # Detect speech timestamps
        timestamp = timing.detect_speech_timestamps(recording_path)
        timestamp = next((entry['start'] for entry in timestamp if 'start' in entry), "None")

        # Transcribe audio to text with error handling
        try:
            transcribed_text = transcribe.speech_to_text(input_path=recording_path, lang=lang).lower()
        except sr.UnknownValueError:
            transcribed_text = "None"

        # Display the transcribed text
        transcription_placeholder.markdown(f"<h3 style='text-align: center;'>{transcribed_text}</h3>",
                                           unsafe_allow_html=True)

        # Store the transcription and timestamp
        if word not in latencies:
            latencies[word] = {}
        latencies[word][transcribed_text] = timestamp

        # Clear placeholders simultaneously before moving to the next word
        time.sleep(1)  # Adjust delay as necessary
        word_placeholder.empty()
        transcription_placeholder.empty()

        # Simulate delay for demonstration purposes (optional)
        time.sleep(1)

    # Clear any remaining placeholders after the loop ends
    word_placeholder.empty()
    transcription_placeholder.empty()

    # Step 4: Calculate distances between transcriptions and the corresponding words
    distances = {}
    for word in words_vector:
        distances[word] = {}
        for transcribed_text in latencies[word]:
            if transcribed_text == "None":
                distance_value = "None"
            else:
                distance_value = distance.calculate_word_distance(transcribed_text, word)
            distances[word][transcribed_text] = distance_value

    return distances, latencies


# Streamlit UI
st.title("Audio Processing and Word Distance Calculation")

st.write("""
    ## Introduction
    This app records audio, detects speech timestamps, transcribes the audio to text, and calculates word distances.
""")

# Define the words vector and recording path template
recording_path_template = "/Users/dennisklose/PycharmProjects/decodePSY/recording{}.wav"

words_vector = create_words_vector("/Users/dennisklose/wf_mod3.txt")

if st.button('Start Processing'):
    distances, latencies = process_audio_and_calculate_distances(recording_path_template, words_vector)

    # Prepare distances table
    distances_table = []
    for word, dist_dict in distances.items():
        for transcribed_text, distance_value in dist_dict.items():
            distances_table.append({'Word': word, 'Response': transcribed_text, 'Distance': distance_value})

    # Prepare latencies table
    latencies_table = []
    for word, lat_dict in latencies.items():
        for transcribed_text, timestamp in lat_dict.items():
            latencies_table.append({'Word': word, 'Response': transcribed_text, 'Latency': timestamp})

    # Merge distances_table and latencies_table based on 'Word' and 'Response'
    combined_table = []
    for dist_row in distances_table:
        for lat_row in latencies_table:
            if dist_row['Word'] == lat_row['Word'] and dist_row['Response'] == lat_row['Response']:
                combined_table.append({
                    'Word': dist_row['Word'],
                    'Response': dist_row['Response'],
                    'Distance': dist_row['Distance'],
                    'Latency': lat_row['Latency']
                })
                break

    # Convert combined_table to pandas DataFrame
    df = pd.DataFrame(combined_table)

    # Display the combined table with left-aligned text
    st.write("### Combined Table")
    st.table(df)