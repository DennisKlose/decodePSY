import streamlit as st
import record
import timing
import transcribe
import distance
import speech_recognition as sr
import matplotlib.pyplot as plt
import ast
import time
import pandas as pd
import google.generativeai as palm
import gensim.downloader as api

palm.configure(api_key="AIzaSyAAVktcnKOUwemyDNDMR7L24MvSLGrKjyE")

@st.cache_resource()
def load_word2vec_model():
    # Load the model
    model = api.load("word2vec-google-news-300")
    return model

# Load the model only once
model = load_word2vec_model()

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

def process_audio_and_calculate_distances(recording_path_template, words_vector, lang="en-US", model=model):
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
    progress_placeholder = st.sidebar.empty()

    # Step 1 to 3: Record, timestamp, and transcribe audio for each word in words_vector
    for idx, word in enumerate(words_vector):
        # Display the current word
        progress_placeholder.markdown(f"<h3 style='text-align: center;'>Progress: {idx + 1}/{len(words_vector)}</h3>",
                                      unsafe_allow_html=True)
        word_placeholder.markdown(f"<h1 style='text-align: center;'>{word}</h1>", unsafe_allow_html=True)

        recording_path = recording_path_template.format(idx)

        # Record audio
        record.record_audio(recording_path)

        # Detect speech timestamps
        timestamp = timing.detect_speech_timestamps(recording_path)
        timestamp = next((entry['start'] for entry in timestamp if 'start' in entry), None)

        # Transcribe audio to text with error handling
        try:
            transcribed_text = transcribe.speech_to_text(input_path=recording_path, lang=lang).lower()
            transcribed_text = clean_transcription(transcribed_text)  # Clean the transcribed text
        except sr.UnknownValueError:
            transcribed_text = None

        # Display the transcribed text
        transcription_placeholder.markdown(f"<h3 style='text-align: center;'>{transcribed_text}</h3>",
                                           unsafe_allow_html=True)

        # Store the transcription and timestamp
        if word not in latencies:
            latencies[word] = {}
        latencies[word][transcribed_text] = timestamp

        # Simulate delay for demonstration purposes (optional)
        time.sleep(2)

        # Clear placeholders simultaneously before moving to the next word
        word_placeholder.empty()
        transcription_placeholder.empty()

    # Clear placeholders after processing all words
    word_placeholder.empty()
    transcription_placeholder.empty()
    progress_placeholder.empty()

    # Step 4: Calculate distances between transcriptions and the corresponding words
    distances = {}
    for word in words_vector:
        distances[word] = {}
        for transcribed_text in latencies[word]:
            if transcribed_text == None:
                distance_value = None
            else:
                distance_value = distance.calculate_word_distance(transcribed_text, word, model)
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

def generate_emotional_vector(model_name, prompt):
    completion = palm.generate_text(
        model=model_name,
        prompt=prompt,
        temperature=1,
        max_output_tokens=4000,
    )

    words_vector = completion.result

    # Find the starting and ending indices of the square brackets
    start_index = words_vector.index('[')
    end_index = words_vector.rindex(']') + 1

    # Extract the content including the square brackets
    words_vector = words_vector[start_index:end_index]

    # Convert the string to a Python list
    words_vector = ast.literal_eval(words_vector)

    return words_vector

model_name = "models/text-bison-001"
prompt = "create a python vector of 21 single everyday words that have deep psychological significance from a psychological perspective. These words should be common and familiar, but should also evoke themes related to the subconscious, dreams, inner conflicts, archetypes, and hidden desires. Each word should be easily understandable and suitable for word association tasks used to explore the subconscious mind. Don't add anything to the code (no print function) and don't repeat the words in the vector."
words_vector = generate_emotional_vector(model_name, prompt)

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

    df.replace('None', None, inplace=True)
    df_cleaned = df.dropna(subset=['Distance', 'Latency'])
    st.write("### Scatter Plot of Latency vs Distance")
    fig, ax = plt.subplots()
    ax.scatter(df_cleaned['Distance'], df_cleaned['Latency'], alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Latency')
    ax.set_title('Latency vs Distance')
    st.pyplot(fig)
