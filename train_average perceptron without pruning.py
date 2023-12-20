#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['bias'] = 1
    return v
    
def test(devfile, model):
    tot, err = 0, 0
    wrong_positive_examples = []
    wrong_negative_examples = []
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        features = make_vector(words)
        prediction = model.dot(features)
        if label * prediction <= 0:
            err += 1
            if label == -1:
                wrong_positive_examples.append((words, prediction))
            elif label == 1:
                wrong_negative_examples.append((words, prediction))


    wrong_positive_examples.sort(key=lambda x: x[1], reverse=True)
     
    wrong_negative_examples.sort(key=lambda x: x[1]) 

      # Display the results
    print("5 Negative Examples that Model Strongly Believes to be Positive:")
    for words, confidence in wrong_positive_examples[:5]:
        print("Words:", words, "Confidence:", confidence)

    print("\n5 Positive Examples that Model Strongly Believes to be Negative:")
    for words, confidence in wrong_negative_examples[:5]:
        print("Words:", words, "Confidence:", -confidence)


        #err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model_avg = svector()
    c = 0
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_avg += label * sent * c
            c = c+1
        model_output = c * model - model_avg
       
        dev_err = test(devfile, model_output)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

     # Sort the features by weight
    sorted_features = sorted(model_output.items(), key=lambda x: x[1], reverse=True)
    
    # Print the top 20 most positive features
    #print("\nTop 20 most positive features:")
    #for feature, weight in sorted_features[:20]:
      #  print(f"{feature}: {weight}")
    
    # Print the top 20 most negative features
   # print("\nTop 20 most negative features:")
    #for feature, weight in sorted_features[-20:]:
       # print(f"{feature}: {weight}")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], 10)
