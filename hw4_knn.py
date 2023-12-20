import numpy as np
from gensim.models import KeyedVectors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

def compute_sentence_embedding(sentence, model):
    words = sentence.split()
    valid_words = [word for word in words if word in model.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model[word] for word in valid_words], axis=0)

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

def create_embeddings(sentences, model):
    return np.array([compute_sentence_embedding(sentence, model) for sentence in sentences])

# Load the custom pre-trained Word2Vec model
model_path = 'embs_train.kv'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found.")
model = KeyedVectors.load(model_path)

# Load the training and development data and calculate positive rates
try:
    train_sentences, train_labels, train_positive_rate = load_data_and_calculate_positive_rate('train.txt')
    dev_sentences, dev_labels, dev_positive_rate = load_data_and_calculate_positive_rate('dev.txt')
    print(f"Training Data Positive Rate: {train_positive_rate:.2f}")
    print(f"Development Data Positive Rate: {dev_positive_rate:.2f}")
except FileNotFoundError as e:
    print(e)
    exit()

# Compute embeddings
train_embeddings = create_embeddings(train_sentences, model)
dev_embeddings = create_embeddings(dev_sentences, model)

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

# Print or store the error rates
for k, error_rate in error_rates:
    print(f"k = {k}, Error Rate = {error_rate}")
