from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Function to read the dataset files
def read_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [line.strip().split('\t') for line in file.readlines()]
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []

# Function to convert sentences to average word embeddings
def sentence_to_avg_embedding(sentence, embeddings):
    words = sentence.split()
    embeddings_list = [embeddings[word] for word in words if word in embeddings]
    return np.mean(embeddings_list, axis=0) if embeddings_list else np.zeros(embeddings.vector_size)

# Preparing the datasets
def prepare_dataset(data, embeddings):
    X, y = [], []
    for label, sentence in data:
        X.append(sentence_to_avg_embedding(sentence, embeddings))
        y.append(1 if label == '+' else 0)
    return np.array(X), np.array(y)

# Paths to the dataset files and embeddings
train_file_path = 'train.txt'
dev_file_path = 'dev.txt'
embeddings_file_path = 'embs_train.kv'

# Load the datasets
train_data = read_dataset(train_file_path)
dev_data = read_dataset(dev_file_path)

# Load the word embeddings
word_embeddings = KeyedVectors.load(embeddings_file_path, mmap='r')

# Prepare training and development datasets
X_train, y_train = prepare_dataset(train_data, word_embeddings)
X_dev, y_dev = prepare_dataset(dev_data, word_embeddings)

# Train the logistic regression model with increased max iterations
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on development dataset and calculate accuracy
dev_predictions = log_reg.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)

# Print development accuracy and error
print(f"Development Accuracy: {dev_accuracy * 100:.2f}%")
print(f"Development Error: {(1 - dev_accuracy) * 100:.2f}%")
