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
gdir = "../gsPvecs/"

if not os.path.exists(gdir):
    os.system("mkdir " + gdir)
    print("new directory created")

allDocs = []
numEpochs = 5
Document = namedtuple("Document", "words tags")
cores = multiprocessing.cpu_count()



# with open(wdir + "guten_sents7.txt", "rb") as allData:
with open(wdir + "doc1.txt", "rb") as allData:
	# contents = allData.read().decode("latin-1")
	for lineNo, line in enumerate(allData):
		try:
			tokens = gs.utils.to_unicode(line).split()
			words = tokens[1:]
			tags = [lineNo]
			allDocs.append(Document(words, tags))
		except:
			print("failed!")
			continue
		
# model1 = Doc2Vec.load(gdir+"ixchelgs.doc2vec")

docList = allDocs

model1 = Doc2Vec(dm=1, dm_mean=1, size=300, window=4, max_vocab_size=60000, workers=cores)

for epoch in range(numEpochs):
    model1.alpha = 0.5
    model1.train(docList)
    print("epoch: %d" % (epoch))
    model1.save(gdir + "doc1.doc2vec", separately=None)
	

# for i in range(2):
#     doc_id = i+1
#     sims = model1.docvecs.most_similar(doc_id, topn=model1.docvecs.count)
#     print(u'TARGET (%d): %s->  \n' % (doc_id, ' '.join(allDocs[doc_id].words)))
    
#     for label, index in [('MOST', 0), ('second most', 1), ('third most', 2)]:
# 	print(u'%s %s: %s ->\n' % (label, sims[index], ' '.join(allDocs[sims[index][0]].words)))
		
		
