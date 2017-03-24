from __future__ import print_function
import math
import numpy as np
import os
import random
import tensorflow as tf
import gensim as gs
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
# from io import open
from collections import namedtuple


wdir = "./testInputs/"
gdir = "./gsPvecs/"

if not os.path.exists(gdir):
	os.system("sudo mkdir " + gdir)
	print("new directory created")

allDocs = []
numEpochs = 2
Document = namedtuple("Document", "words tags")
cores = multiprocessing.cpu_count()


lineNo = 0

def lineToWords(line):
	global lineNo
	tokens = gs.utils.to_unicode(line).split()
	words = tokens[1:]
	tags = [lineNo]
	lineNo += 1
	return Document(words, tags)

class Corpus(object):
	def __init__(self, idx):
		self.filename = wdir + "doc" + str(idx) + ".txt"
	def __iter__(self):
		for line in open(self.filename, "rb"):
			yield lineToWords(line)

model1 = Doc2Vec(dm=1, dm_mean=1, size=300, window=4, max_vocab_size=60000, negative=5, hs=0, min_count=1, workers=cores)
docList = allDocs[:]
docList = docList

for i in range(3):
	itr = Corpus(i)
	if i >= 1:
		update = True
	else:
		update = False
	print(i)
	model1.build_vocab(itr, update=update)

# print("number of training sentences: %d" % (len(docList)))

print(model1.docvecs.count)

for epoch in range(numEpochs):
	model1.alpha = 0.5
	for i in range(3):
		itr = Corpus(i)
		model1.train(itr)
		print(i)
	print("epoch: %d" % (epoch))
	
model1.save(gdir + "gw1000.doc2vec", separately=None)

print(model1.docvecs[0])
print(model1.docvecs.count)

# for i in range(2):
# 	doc_id = np.random.randint(model1.docvecs.count)
# 	sims = model1.docvecs.most_similar(doc_id, topn=model1.docvecs.count)
# 	print(u'TARGET (%d): %s->  \n' % (doc_id, ' '.join(allDocs[doc_id].words)))
	
# 	for label, index in [('MOST', 0), ('second most', 1), ('third most', 2)]:
# 		print(u'%s %s: %s ->\n' % (label, sims[index], ' '.join(allDocs[sims[index][0]].words)))
		
		

