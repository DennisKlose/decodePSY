import gensim.downloader as api
import numpy as np

def calculate_word_distance(word1, word2):
    """
    Calculates the Euclidean distance between two words in a pre-trained Word2Vec model.
    """
    try:
        # Load the pre-trained Word2Vec model
        model = api.load('word2vec-google-news-300')

        # Get word vectors
        vector1 = model[word1]
        vector2 = model[word2]

        # Calculate Euclidean distance
        euclidean_distance = np.linalg.norm(vector1 - vector2)

        return euclidean_distance

    except KeyError:
        # Handle case where either word1 or word2 is not present in the model
        return "None"

    except Exception as e:
        # Handle any other exceptions that may occur (e.g., model loading issues)
        print(f"Exception occurred: {e}")
        return None
