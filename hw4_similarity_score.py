import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

def compute_sentence_embedding(sentence, model):
    words = sentence.split()
    valid_words = [word for word in words if word in model.key_to_index]
    if not valid_words:  # Handle case where no words are in the model's vocabulary
        return np.zeros(model.vector_size)
    return np.mean([model[word] for word in valid_words], axis=0)

def load_sentences(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def find_closest_sentence(target_sentence, sentences, model):
    target_embedding = compute_sentence_embedding(target_sentence, model)
    max_similarity = -1
    closest_sentence = None

    for full_sentence in sentences:
        # Extract the text part of the sentence, excluding the label
        sentence_text = full_sentence.strip('+-\t\n')
        if sentence_text != target_sentence:  # Exclude the target sentence itself
            embedding = compute_sentence_embedding(sentence_text, model)
            similarity = cosine_similarity([target_embedding], [embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                closest_sentence = full_sentence

    return closest_sentence, max_similarity


# Load the custom pre-trained Word2Vec model
model = KeyedVectors.load('embs_train.kv')

# Load your training set sentences
file_path = 'train.txt'  # Replace with your file path
sentences = load_sentences(file_path)

# Specify your target sentence
target_sentence = "places a slightly believable love triangle in a difficult to swallow setting , and then disappointingly moves the story into the realm of an improbable thriller"  # Replace with the sentence you want to compare

# Find the closest sentence
closest_sentence, similarity = find_closest_sentence(target_sentence, sentences, model)

print("Closest sentence:", closest_sentence)
print("Similarity:", similarity)
