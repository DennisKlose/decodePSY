import torch
from torch.hub import load as load_model
import os

def detect_speech_timestamps(file_path, sampling_rate=16000, repo_or_dir='snakers4/silero-vad', model='silero_vad'):
    """
    Detects speech timestamps in an audio file.
    """
    # Set the number of threads for Torch
    torch.set_num_threads(1)

    # Load the VAD model
    model, utils = load_model(repo_or_dir=repo_or_dir, model=model)
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Read the audio file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    wav = read_audio(file_path, sampling_rate=sampling_rate)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, visualize_probs=True, return_seconds=True)
    return speech_timestamps



