import numpy as np
import os
import time

class SVector(dict):
    def __init__(self):
        super().__init__()

    def dot(self, other):
        # Dot product calculation without the bias term
        return sum(self.get(k, 0) * v for k, v in other.items())

def build_vocabulary(data_file):
    vocab = {}
    with open(data_file, 'r') as file:
        for line in file:
            _, sentence = line.strip().split(maxsplit=1)
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab

def compute_sentence_one_hot(sentence, vocab):
    words = sentence.split()
    sentence_vector = np.zeros(len(vocab))
    for word in words:
        if word in vocab:
            sentence_vector[vocab[word]] = 1
    return sentence_vector

def load_data(file_path, vocab):
    sentences, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            label, sentence = line.strip().split(maxsplit=1)
            labels.append(1 if label == '+' else -1)
            sentences.append(compute_sentence_one_hot(sentence, vocab))
    return np.array(sentences), np.array(labels)

def train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=10):
    t = time.time()
    best_err = 1.
    model = SVector()
    for it in range(1, epochs + 1):
        updates = 0
        for i in range(len(train_embeddings)):
            label = train_labels[i]
            sent_embedding = dict(enumerate(train_embeddings[i]))
            if label * (model.dot(sent_embedding)) <= 0:
                updates += 1
                update = {k: label * v for k, v in sent_embedding.items()}
                model.update((k, model.get(k, 0) + v) for k, v in update.items())

        dev_err = test(dev_embeddings, dev_labels, model)
        best_err = min(best_err, dev_err)
        print(f"epoch {it}, update {updates / len(train_embeddings) * 100:.1f}%, dev {dev_err * 100:.1f}%")
    print(f"best dev err {best_err * 100:.1f}%, |w|={len(model)}, time: {time.time() - t:.1f} secs")

def test(dev_embeddings, dev_labels, model):
    correct = 0
    for i in range(len(dev_embeddings)):
        sent_embedding = dict(enumerate(dev_embeddings[i]))
        prediction = 1 if model.dot(sent_embedding) > 0 else -1
        if prediction == dev_labels[i]:
            correct += 1
    return 1 - correct / len(dev_embeddings)

# Create Vocabulary from training data
vocab = build_vocabulary('train.txt')

# Load the training and development data with one-hot encoding
train_embeddings, train_labels = load_data('train.txt', vocab)
dev_embeddings, dev_labels = load_data('dev.txt', vocab)

# Train the model
train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=10)
