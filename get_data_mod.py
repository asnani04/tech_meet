import numpy as np
import re
import itertools
from collections import Counter, OrderedDict
import glob
import os
import pickle as pkl

def clean_str(string):
    string = re.sub(r"[^A-Za-z']", " ", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r"\ +", " ", string)
    return string.strip().lower()

def build_docs(path):
    currdir = os.getcwd()
    docs = []
    for filename in os.listdir(path):
        try:
            # print(path + filename)
            with open(path + filename, "r") as f:
                doc = f.read()
                docs.append(doc)
                # print(filename + " done")
        except:
            # print(filename + " not found")
            pass
    print("number of docs: %d " % (len(docs)))
    return docs

def build_data(sentences):
    x_text = [clean_str(sent) for sent in sentences]
    # print(x_text)
    x_text = [s.split(" ") for s in x_text]
    # for strings in x_text:
    #     if strings == " ":
    #         x_text.remove(string)
    return x_text


# Pad all sentences to maximum length

def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    
    padded_sentences = []
    count = 0
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        # print(sentence[:5])
        count = count + len(sentence)
        # print(len(sentence))
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    print(float(count/len(sentences)))
    return padded_sentences

def build_dataset(sentences_train, vocabulary_size):
    count = [['UNK', -1]]
    sentences = []
    for sentence in sentences_train:
        sentences.append(sentence)
        
    count.extend(Counter(itertools.chain(*sentences)).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data_train = list()
    data_test = list()
    unk_count = 0
    for ind, sentence in enumerate(sentences_train):
        data_train.append([])
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data_train[ind].append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data_train, data_test, count, dictionary, reverse_dictionary


def load_data(vocab_size):
    #Load data from files

    path = "../toyset/"
    docs = []
    docs_train = build_docs(path)
    x_train = build_data(docs_train)
    maxlen = 60
    docs_padded_train = pad_sentences(x_train, maxlen)
    # print(docs_padded_train[:5])
    data_train, data_test, count, vocabulary, vocabulary_inv = build_dataset(docs_padded_train, vocab_size)
    # print(vocabulary['a'])
    data_train = np.array(data_train)
    np.savez("../toydata.npz", data=data_train)
    pkl.dump(vocabulary, open("./toyvocab.p", "wb"))
    pkl.dump(vocabulary_inv, open("./toyvocab_inv.p", "wb"))
    # print(data_train[0])
    return [data_train, docs_padded_train, vocabulary, vocabulary_inv]


# load_data(1000)
