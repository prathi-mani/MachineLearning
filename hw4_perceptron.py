import numpy as np
import os
import time
from gensim.models import KeyedVectors

class SVector(dict):
    def __init__(self):
        super().__init__()

    def dot(self, other):
        # Dot product calculation without the bias term
        return sum(self.get(k, 0) * v for k, v in other.items())

def compute_sentence_embedding(sentence, model):
    words = sentence.split()
    valid_words = [word for word in words if word in model.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model[word] for word in valid_words], axis=0)

def load_data(file_path, model):
    sentences, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            label, sentence = line.strip().split(maxsplit=1)
            labels.append(1 if label == '+' else -1)
            sentences.append(compute_sentence_embedding(sentence, model))
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

# Load the custom pre-trained Word2Vec model
model_path = 'embs_train.kv'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found.")
model = KeyedVectors.load(model_path)

# Load the training and development data
train_embeddings, train_labels = load_data('train.txt', model)
dev_embeddings, dev_labels = load_data('dev.txt', model)

# Train the model
train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=10)
