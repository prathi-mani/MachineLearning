import numpy as np
import os
import time
from gensim.models import KeyedVectors

class SVector(dict):
    def dot(self, other):
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


def train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=5):
    t = time.time()
    best_err = 1.
    weights = SVector()  # Current weights
    total_weights = SVector()  # Sum of weights over time
    n_examples = 0

    for epoch in range(epochs):
        updates = 0
        for i, sent_embedding in enumerate(train_embeddings):
            n_examples += 1
            label = train_labels[i]
            sent_embedding = dict(enumerate(sent_embedding))
            if label * weights.dot(sent_embedding) <= 0:
                updates += 1
                for k, v in sent_embedding.items():
                    weights[k] = weights.get(k, 0) + label * v
                    # Directly accumulate the weights after each example
                    total_weights[k] = total_weights.get(k, 0) + weights[k]

        # Calculate the averaged weights outside the inner loop, after all examples have been seen
        averaged_weights = {k: v / n_examples for k, v in total_weights.items()}

        dev_err = test(dev_embeddings, dev_labels, averaged_weights)
        best_err = min(best_err, dev_err)
        print(f"Epoch {epoch+1}, updates {updates / len(train_embeddings) * 100:.2f}%, dev error {dev_err * 100:.2f}%")

    print(f"Best dev error {best_err * 100:.2f}%, |w|={len(averaged_weights)}, time: {time.time() - t:.2f} seconds")

    # Return the final averaged weights for potential use outside the training function
    return averaged_weights




def test(dev_embeddings, dev_labels, model):
    correct = 0
    for i in range(len(dev_embeddings)):
        sent_embedding = dict(enumerate(dev_embeddings[i]))
        prediction = 1 if sum(model.get(k, 0) * v for k, v in sent_embedding.items()) > 0 else -1
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
