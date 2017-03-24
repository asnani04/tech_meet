import numpy as np
import re
import itertools
from collections import Counter, OrderedDict
import glob
import os
import cPickle as pkl

# Function to clean strings - remove every character except alphabets

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9']", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"br$", " ", string)
    return string.strip().lower()

# Function to get all the training data in python lists

def build_sents(path, test=False):
	currdir = os.getcwd()
	sentences = []
	labels = []
	if test == True:
		filename = "./enron_test.txt"
	else:
		filename = "./enron_train.txt"
	with open(filename, 'r') as f:
		lines = f.readlines()
		f.seek(0)
		for line in lines:
			label, sent = line.split('\t')
			if label == 'Yes':
				labels.append([1, 0])
			else:
				labels.append([0, 1])
			sentences.append(sent)
	os.chdir(currdir)
	print(len(sentences))
	return sentences, labels


def build_data(sentences):
    x_text = [clean_str(sent) for sent in sentences]
    x_text = [s.split(" ") for s in x_text]
    return x_text

# Pad all sentences to maximum length

def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    
    padded_sentences = []
    count = 0
    # for i in range(100):
        # print(len(sentences[i]))
    for i in range(len(sentences)):
        sentence = sentences[i]
        count = count + len(sentence)
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    print(float(count/len(sentences)))
    return padded_sentences


def build_dataset(sentences_train, sentences_test, vocabulary_size):
	count = [['UNK', -1]]
	sentences = []
	for sentence in sentences_train:
		sentences.append(sentence)
	for sentence in sentences_test:
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
	for ind, sentence in enumerate(sentences_test):
		data_test.append([])
		for word in sentence:
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0  # dictionary['UNK']
				unk_count = unk_count + 1
			data_test[ind].append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data_train, data_test, count, dictionary, reverse_dictionary

# This function uses pre built vocabulary to build our dataset

def build_dataset_new(sentences_train, sentences_test, vocabulary_size):
	vocabulary = pkl.load(open("./vocab.p", "rb"))
	dictionary = vocabulary
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
	for ind, sentence in enumerate(sentences_test):
		data_test.append([])
		for word in sentence:
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0  # dictionary['UNK']
				unk_count = unk_count + 1
			data_test[ind].append(index)
	count = [['unk', -1]]
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	return data_train, data_test, count, dictionary, reverse_dictionary


def load_data_and_labels(vocab_size):
    # Load data from files
    if os.path.exists("./enron_train2.npz") and os.path.exists("./enron_test2.npz"):
        print("loading saved data")
        z_train = np.load("./enron_train1.npz")
        z_test = np.load("./enron_test1.npz")
        data_train = z_train['data']
        data_test = z_test['data']
        y_train = z_train['y']
        y_test = z_test['y']
        vocabulary = pkl.load(open("./vocab.p", "rb"))
        vocabulary_inv = pkl.load(open("./vocab_inv.p", "rb"))
        sentences_padded_train = pkl.load(open("./sents_padded.p", "wb"))
    else:
        path = "./"
        sentences = []
        labels = []
        sentences_train, train_labels = build_sents(path, test=False)
        sentences_test, test_labels = build_sents(path, test=True)

        x_train = build_data(sentences_train)
        x_test = build_data(sentences_test)
        maxlen_train = max(len(x) for x in x_train)
        maxlen_test = max(len(x) for x in x_test)
        maxlen = max(maxlen_train, maxlen_test)
        print("maxlen: %d", maxlen)
        sentences_padded_train = pad_sentences(x_train, maxlen)
        sentences_padded_test = pad_sentences(x_test, maxlen)
        data_train, data_test, count, vocabulary, vocabulary_inv = build_dataset_new(sentences_padded_train, sentences_padded_test, vocab_size)

        #print(len(data))
        #print(len(data[1]))
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        # x, y = build_input_data(sentences_padded, labels, vocabulary)
        #print(data[:2])
        #print(len(vocabulary))
        np.savez("./enron_train1.npz", data=data_train, y=y_train)
        np.savez("./enron_test1.npz", data=data_test, y=y_test)
        pkl.dump(vocabulary, open("./vocab.p", "wb"))
        pkl.dump(vocabulary_inv, open("./vocab_inv.p", "wb"))
        pkl.dump(sentences_padded_train, open("./sents_padded.p", "wb"))
    return [data_train, data_test, y_train, y_test, sentences_padded_train, vocabulary, vocabulary_inv]

#load_data_and_labels(45)
