import sys
import nltk
import numpy
from scipy import spatial
from nltk.corpus import stopwords
from collections import Counter
import math
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import json
#import matplotlib.pyplot as plt
# from aligner import *

def getIDF(*args):
	docCount = 0.
	wordCount = Counter()
	for sents in args:
		for sent in sents:
			wordCount += Counter(list(set(sent)))
			docCount += 1.
	# calculate idf
	idf = {}
	for word, count in wordCount.iteritems():
		idf[word] = math.log(docCount / count)
	return idf

def read_sentences(subset):
	sentences_A = []
	sentences_B = []
	with open(subset) as inp_hndl:
		for example in inp_hndl:
			A, B = example.strip().split('\t')
			sentence_A = nltk.word_tokenize(A.lower().decode('utf8'))
			sentence_B = nltk.word_tokenize(B.lower().decode('utf8'))

			sentences_A.append(sentence_A)
			sentences_B.append(sentence_B)

	# estimate IDF
	idfDict = getIDF(sentences_A, sentences_B)

	return sentences_A, sentences_B, idfDict

# def word_alignment(sentence_A, sentence_B):
# 	alignments = align(sentence_A, sentence_B)
# 	countWordA = set(sentence_A)
# 	countWordB = set(sentence_B)
# 	countAlignWordA = set([])
# 	countAlignWordB = set([])
# 	for a in alignments[0]:
# 		countAlignWordA.add(a[0])
# 		countAlignWordB.add(a[1])
# 	sim = float(len(countAlignWordA)+len(countAlignWordB)) / (len(countWordA)+len(countWordB))

def cosine_similarity(listA, listB):
	result = 1 - spatial.distance.cosine(listA, listB)
	return result

def dictSim_NON(dictA,dictB):
	ans = 0.
	for key,value in dictA.iteritems():
		ans += value*dictB.get(key,0)
	return ans

def getNgramPatterns(L,N=2):
	patterns = {}
	count = len(L)-N+1
	for j in xrange(count):
		x = tuple(L[j:j+N])
		if x in patterns:
			patterns[x]+=1
		else:
			patterns[x]=1.0
	return patterns,count

def sentence_vector_similarity(A, B, embeddings, idfDict):
	# get dimension
	dimensions = len(embeddings[embeddings.keys()[0]])

	features = []
	for i in range(len(A)):

		# feature 1: Macro Information Distance
		vectorA = numpy.zeros(dimensions)
		vectorB = numpy.zeros(dimensions)
		for wordA in A[i]:
			if wordA in embeddings:
				vectorA += embeddings[wordA]*idfDict.get(wordA, 0.)
		for wordB in B[i]:
			if wordB in embeddings:
				vectorB += embeddings[wordB]*idfDict.get(wordB, 0.)
		if numpy.linalg.norm(vectorA) == 0 or numpy.linalg.norm(vectorB) == 0:
			cos_s = 0.
		else:
			cos_s = cosine_similarity(vectorA, vectorB)

		# feature 2: word alignment
		# try:
		# 	sim = word_alignment(A[i], B[i])
		# except:
		# 	print "error!"
		# 	sim = 0.5
		A_Ngram_CHR,A_count = getNgramPatterns(''.join(A[i]),N=3)
		B_Ngram_CHR,B_count = getNgramPatterns(''.join(B[i]),N=3)
		x = dictSim_NON(A_Ngram_CHR,B_Ngram_CHR)/(A_count+B_count) if (A_count+B_count) else 0
		# y1 = x/A_count if A_count else 0
		# y2 = x/B_count if B_count else 0
		f = [cos_s,x,1]
		# A_Ngram_WRD,A_count = getNgramPatterns(A[i])
		# B_Ngram_WRD,B_count = getNgramPatterns(B[i])
		# x = dictSim_NON(A_Ngram_WRD,B_Ngram_WRD)/(A_count+B_count) if (A_count+B_count) else 0
		# f.append(x)

		features.append(f)
		# features.append([sim])

	return features

def training():
	train_gs = ["pair/STS.gs.crosslingual-trial.txt"]
# "train/STS2012-en-test/STS.gs.MSRpar.txt"

	train_input = ["pair/dev.STS.input.crosslingual.txt"]
# ["train/STS2012-en-test/STS.input.MSRpar.txt",
	features = []
	labels = []
	for i in range(len(train_input)):
		sentencesA, sentencesB, idfDict = read_sentences(train_input[i])
		features += sentence_vector_similarity(sentencesA, sentencesB, embeddings, idfDict)

		# read gold standard
		with open(train_gs[i], "rb") as f:
			labels += map(float, f.read().strip().split())
	assert len(features) == len(labels)

	# train model
	model = MLPRegressor(hidden_layer_sizes = (30,30), max_iter = 10000,
										activation = 'logistic')
	# params = {'n_estimators': 10, 'max_depth': 6,
 #        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
	model.fit(features, labels)
	return model

if __name__ == "__main__":
	global embeddings
	embeddings = {}
	print "start reading"
	# with open('paragram_vectors.txt') as pretrained:
	with open('embeddings/paragram-phrase-XXL.txt') as pretrained:
		for line in pretrained:
			wordL = line.strip().split()
			for i in xrange(1,len(wordL)):
				wordL[i] = float(wordL[i])
			embeddings[wordL[0]] = numpy.array(wordL[1:])
	print "size: %d"%len(embeddings)
	
	with open('embeddings/paragram_300_sl999.txt') as pretrained:
		for line in pretrained:
			wordL = line.strip().split()
			for i in xrange(1,len(wordL)):
				wordL[i] = float(wordL[i])
			if(wordL[0] not in embeddings):
				embeddings[wordL[0]] = numpy.array(wordL[1:])
	print "size: %d"%len(embeddings)
	
	print "finish reading"
	
	print "start training"
	model = training()

	test_input = ["pair/test.STS.input.multisource.txt","pair/test.STS.input.news.txt"]
	test_output = ["testoutput/test.STS.output.multisource.txt","testoutput/test.STS.output.news.txt"]

	print "start testing"

	for i in range(len(test_input)):
		print "processing %s"%(test_input[i])
		sentencesA, sentencesB, idfDict = read_sentences(test_input[i])
		features = sentence_vector_similarity(sentencesA, sentencesB, embeddings, idfDict)
		scores = model.predict(features)
		results = map(str, scores)
		with open(test_output[i], "wb") as f:
			f.write("\n".join(results))
