import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def build_vocabulary(sentences):
    vocabulary = set()
    for sentence in sentences:
        words = sentence.split()
        vocabulary.update(words)
    return {word: i for i, word in enumerate(vocabulary)}

def compute_sentence_embedding(sentence, vocabulary):
    words = sentence.split()
    sentence_vector = np.zeros(len(vocabulary))
    valid_word_count = 0
    for word in words:
        if word in vocabulary:
            word_index = vocabulary[word]
            word_vector = np.zeros(len(vocabulary))
            word_vector[word_index] = 1
            sentence_vector += word_vector
            valid_word_count += 1
    return sentence_vector / valid_word_count if valid_word_count > 0 else np.zeros(len(vocabulary))

def load_data_and_calculate_positive_rate(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    sentences = []
    labels = []
    positive_count = 0
    total_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            label, sentence = parts
            sentences.append(sentence)
            is_positive = label == '+'
            labels.append(1 if is_positive else 0)
            positive_count += is_positive
            total_count += 1
    positive_rate = positive_count / total_count if total_count > 0 else 0
    return sentences, labels, positive_rate

# Load the training and development data and calculate positive rates
train_sentences, train_labels, train_positive_rate = load_data_and_calculate_positive_rate('train.txt')
dev_sentences, dev_labels, dev_positive_rate = load_data_and_calculate_positive_rate('dev.txt')

# Build vocabulary from training sentences
vocabulary = build_vocabulary(train_sentences + dev_sentences)

# Compute embeddings using one-hot vectors
train_embeddings = np.array([compute_sentence_embedding(sentence, vocabulary) for sentence in train_sentences])
dev_embeddings = np.array([compute_sentence_embedding(sentence, vocabulary) for sentence in dev_sentences])

# Iterate over values of k and compute error rates
error_rates = []
for k in range(1, 100, 2):  # k = 1, 3, ..., 99
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_embeddings, train_labels)
    
    # Predict on the development set
    predictions = knn.predict(dev_embeddings)
    
    # Calculate error rate
    error_rate = 1 - accuracy_score(dev_labels, predictions)
    error_rates.append((k, error_rate))

# Print the error rates
for k, error_rate in error_rates:
    print(f"k = {k}, Error Rate = {error_rate:.4f}")
