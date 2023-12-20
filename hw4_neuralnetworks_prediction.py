from gensim.models import KeyedVectors
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Function to read the dataset files
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip().split('\t') for line in file.readlines()]
    return data

# Function to convert sentences to average word embeddings
def sentence_to_avg_embedding(sentence, embeddings):
    words = sentence.split()
    embeddings_list = [embeddings[word] for word in words if word in embeddings]
    if embeddings_list:
        avg_embedding = np.mean(embeddings_list, axis=0)
    else:
        avg_embedding = np.zeros(embeddings.vector_size)  # Zero vector for sentences with no words in embeddings
    return avg_embedding

# Function to prepare the datasets
def prepare_dataset(data, embeddings):
    X, y = [], []
    for label, sentence in data:
        X.append(sentence_to_avg_embedding(sentence, embeddings))
        y.append(1 if label == '+' else 0)  # Convert labels to binary format (+: 1, -: 0)
    return np.array(X), np.array(y)

# Function to predict labels for new sentences
def predict_labels(sentences, embeddings, model):
    X_test = np.array([sentence_to_avg_embedding(sentence, embeddings) for sentence in sentences])
    predictions = model.predict(X_test)
    return ['+' if pred >= 0.5 else '-' for pred in predictions]

# Function to read test file sentences
def read_test_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

# Function to predict labels and update sentences
def update_test_sentences_with_predictions(sentences, embeddings, model):
    updated_sentences = []
    predictions = predict_labels(sentences, embeddings, model)
    for sentence, label in zip(sentences, predictions):
        updated_sentence = sentence.replace('?', label)
        updated_sentences.append(updated_sentence)
    return updated_sentences

# Start the timer
start_time = time.time()

# Paths to the dataset files and embeddings
train_file_path = 'train.txt'
dev_file_path = 'dev.txt'
test_file_path = 'test.txt'  # Replace with your test file path
embeddings_file_path = 'embs_train.kv'

# Load the datasets
train_data = read_dataset(train_file_path)
dev_data = read_dataset(dev_file_path)

# Load the word embeddings
word_embeddings = KeyedVectors.load(embeddings_file_path, mmap='r')

# Prepare training and development datasets
X_train, y_train = prepare_dataset(train_data, word_embeddings)
X_dev, y_dev = prepare_dataset(dev_data, word_embeddings)

# Neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=word_embeddings.vector_size))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_dev, y_dev)
print("Development Accuracy:", accuracy)

# Read test sentences
test_sentences = read_test_sentences(test_file_path)

# Update test sentences with predictions
updated_test_sentences = update_test_sentences_with_predictions(test_sentences, word_embeddings, model)

# Write the updated sentences to a new file
output_file_path = 'updated_test_sentences.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for sentence in updated_test_sentences:
        file.write(sentence + '\n')

print(f"Updated sentences written to {output_file_path}")

# End the timer and print the running time
end_time = time.time()
total_time = end_time - start_time
print(f"Total running time: {total_time} seconds")
