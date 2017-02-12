import sys
import nltk
import numpy
from scipy import spatial
from nltk.corpus import stopwords
from collections import Counter
import math
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import json
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

def get_chunks_POSs(text):
	POS = pos_tag(word_tokenize(text))
	nouns = []
	vbs = []
	for w,pos in POS:
		if len(pos)>1 and pos[:2]=='VB':
			vbs.append(w)
		elif len(pos)>1 and (pos[:2]== "NN" or pos=='PRP'):
			nouns.append(w)
	chunked = ne_chunk(POS)
	prev = None
	current_chunk = []
	continuous_chunk = []
	for i in chunked:
	    if type(i) == Tree:
	        current_chunk += [token for token, _ in i.leaves()]
	    elif current_chunk:
			for named_entity in current_chunk:
				continuous_chunk.append(named_entity)
			current_chunk = []
	    else:
	        continue
	return continuous_chunk,vbs

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

def read_alignments(files):
	aligns = []
	with open(files) as inp_al:
		for digit in inp_al:
			digit = float(digit.strip())
			if(digit == -1):
				digit = 0.2
			elif(digit < 0.1):
				digit = 0
			aligns.append(digit)
	print len(aligns)
	return aligns

def read_sentences(subset, files):
	sentences_A = []
	sentences_B = []
	NEs_A = []
	NEs_B = []
	spw_A = []
	spw_B = []
	with open(subset) as inp_hndl:
		for example in inp_hndl:
			A, B = example.strip().split('\t')
			sentence_A = nltk.word_tokenize(A.lower().decode('utf8'))
			sentence_B = nltk.word_tokenize(B.lower().decode('utf8'))

			sentences_A.append(sentence_A)
			sentences_B.append(sentence_B)
			neA,vbA = get_chunks_POSs(A.decode('utf8'))
			neB,vbB = get_chunks_POSs(B.decode('utf8'))
			NEs_A.append(neA)
			NEs_B.append(neB)
			spw_A.append(vbA)
			spw_B.append(vbB)

	# estimate IDF
	idfDict = getIDF(sentences_A, sentences_B)

	aligns = read_alignments(files)
	assert len(sentences_A) == len(aligns)

	return sentences_A, sentences_B, idfDict, aligns, NEs_A, NEs_B,spw_A,spw_B

def cosine_similarity(listA, listB):
	result = 1 - spatial.distance.cosine(listA, listB)
	return result

def JaccardSim(sA,sB):
	setA = set(sA)
	setB = set(sB)
	ans = 0.
	for w in setA:
		if w in setB:
			ans += 1
	return ans/(len(setA)+len(setB)-ans) if ans>0 else 0

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

def sumEmbeddingSim(lA,lB,embeddings,idfDict,dimensions):
	vectorA = numpy.zeros(dimensions)
	vectorB = numpy.zeros(dimensions)
	for wordA in lA:
		if wordA in embeddings:
			vectorA += embeddings[wordA]*idfDict.get(wordA, 0.)
	for wordB in lB:
		if wordB in embeddings:
			vectorB += embeddings[wordB]*idfDict.get(wordB, 0.)
	if numpy.linalg.norm(vectorA) == 0 or numpy.linalg.norm(vectorB) == 0:
		return 0.
	else:
		return cosine_similarity(vectorA, vectorB)

def sum_maxSim(sA,sB,embeddings,idfDict):
	divisor = divided = 0.
	for wA in sA:
		if wA in embeddings:
			m = 0
			divisor += idfDict.get(wA,0.)
			for wB in sB:
				if wB in embeddings:
					m = max(m,embeddings[wA].dot(embeddings[wB]))
			divided += m*idfDict.get(wA,0.)
	return divided/divisor

def sentence_vector_similarity(A, B, embeddings, idfDict, aligns, NEs_A, NEs_B,spw_A,spw_B):
	# get dimension
	dimensions = len(embeddings[embeddings.keys()[0]])

	features = []
	for i in range(len(A)):
		f = []

		f.append(sumEmbeddingSim(A[i],B[i],embeddings,idfDict,dimensions))

		f.append(sumEmbeddingSim(spw_A[i],spw_B[i],embeddings,idfDict,dimensions))
		f.append(sumEmbeddingSim(spw_A[i],spw_B[i],embeddings,idfDict,dimensions))
		A_Ngram_CHR,A_count = getNgramPatterns(''.join(A[i]),N=3)
		B_Ngram_CHR,B_count = getNgramPatterns(''.join(B[i]),N=3)
		x = dictSim_NON(A_Ngram_CHR,B_Ngram_CHR)/(A_count+B_count) if (A_count+B_count) else 0

		f.append(x)

		A_Ngram_CHR,A_count = getNgramPatterns(''.join(A[i]),N=4)
		B_Ngram_CHR,B_count = getNgramPatterns(''.join(B[i]),N=4)
		x = dictSim_NON(A_Ngram_CHR,B_Ngram_CHR)/(A_count+B_count) if (A_count+B_count) else 0

		A_Ngram_CHR,A_count = getNgramPatterns(A[i],N=2)
		B_Ngram_CHR,B_count = getNgramPatterns(B[i],N=2)
		x = dictSim_NON(A_Ngram_CHR,B_Ngram_CHR)/(A_count+B_count) if (A_count+B_count) else 0

		f.append(aligns[i])
		f.append(sum_maxSim(A[i],B[i],embeddings,idfDict)+sum_maxSim(B[i],A[i],embeddings,idfDict))
		f.append(1)

		features.append(f)

	return features

def training():

	train_gs = ["train/STS2012-en-train/STS.gs.MSRpar.txt", "train/STS2012-en-train/STS.gs.MSRvid.txt",
"train/STS2012-en-train/STS.gs.SMTeuroparl.txt"]

	train_input = ["train/STS2012-en-train/STS.input.MSRpar.txt","train/STS2012-en-train/STS.input.MSRvid.txt",
"train/STS2012-en-train/STS.input.SMTeuroparl.txt"]

	train_align = ["trainalign/2012/STS.alignment.MSRpar.txt", "trainalign/2012/STS.alignment.MSRvid.txt",
"trainalign/2012/STS.alignment.SMTeuroparl.txt"]

	dictionary = Dictionary([])
	features = []
	labels = []
	aligns = []
	for i in range(len(train_input)):
		sentencesA, sentencesB, idfDict, aligns, NEs_A, NEs_B, spw_A, spw_B = read_sentences(train_input[i],train_align[i])
		features += sentence_vector_similarity(sentencesA, sentencesB, embeddings, idfDict, aligns, NEs_A, NEs_B, spw_A, spw_B)
		dictionary.merge_with(Dictionary(sentencesA+sentencesB))
		# read gold standard
		with open(train_gs[i], "rb") as f:
			labels += map(float, f.read().strip().split())
	corpus_A = []
	corpus_B = []
	for i in range(len(train_input)):
		sentencesA, sentencesB, _, _, _, _, _, _ = read_sentences(train_input[i],train_align[i])
		for doc in sentencesA:
			corpus_A.append(dictionary.doc2bow(doc))
		for doc in sentencesB:
			corpus_B.append(dictionary.doc2bow(doc))
	NUM_TPC = 14
	topicModel = LdaModel(corpus_A+corpus_B, num_topics = NUM_TPC)
	assert len(corpus_A)==len(corpus_B)==len(features) == len(labels)
	for i in xrange(len(corpus_A)):

		vectorA = numpy.zeros(NUM_TPC)
		vectorB = numpy.zeros(NUM_TPC)
		for j,prob in topicModel[corpus_A[i]]:
			vectorA[j] = prob
		for j,prob in topicModel[corpus_B[i]]:
			vectorB[j] = prob
		if numpy.linalg.norm(vectorA) == 0 or numpy.linalg.norm(vectorB) == 0:
			features[i].append(0.)
		else:
			features[i].append(cosine_similarity(vectorA, vectorB))

	# train model
	# model = MLPRegressor(hidden_layer_sizes = (100,100), max_iter = 10000,
	# 									activation = 'logistic')
	model = Ridge()
	model.fit(features, labels)
	return model,topicModel, dictionary

if __name__ == "__main__":
	global embeddings
	embeddings = {}
	print "start reading"
	with open('embeddings/paragram-phrase-XXL.txt') as pretrained:
		for line in pretrained:
			wordL = line.strip().split()
			for i in xrange(1,len(wordL)):
				wordL[i] = float(wordL[i])
			embeddings[wordL[0]] = numpy.array(wordL[1:])
	print "size: %d"%len(embeddings)
	# with open('embeddings/paragram_300_sl999.txt') as pretrained:
	# 	for line in pretrained:
	# 		wordL = line.strip().split()
	# 		for i in xrange(1,len(wordL)):
	# 			wordL[i] = float(wordL[i])
	# 		if(wordL[0] not in embeddings):
	# 			embeddings[wordL[0]] = numpy.array(wordL[1:])
	# print "size: %d"%len(embeddings)
	print "finish reading"
	# sentencesA, sentencesB = read_sentences(sys.argv[1])
	# sentence_vector_similarity(sentencesA, sentencesB, embeddings, sys.argv[2])

	print "start training"
	model,topicModel,dictionary = training()

	test_input = [sys.argv[1]]
	test_align = [sys.argv[2]]
	test_output = [sys.argv[3]]

	print "start testing"

	for i in range(len(test_input)):
		print "processing %s"%(test_input[i])
		sentencesA, sentencesB, idfDict, aligns, NEs_A, NEs_B, spw_A, spw_B = read_sentences(test_input[i], test_align[i])
		features = sentence_vector_similarity(sentencesA, sentencesB, embeddings, idfDict, aligns, NEs_A, NEs_B, spw_A, spw_B)
		for k in xrange(len(features)):
			NUM_TPC = 14
			vectorA = numpy.zeros(NUM_TPC)
			vectorB = numpy.zeros(NUM_TPC)
			for j,prob in topicModel[dictionary.doc2bow(sentencesA[k])]:
				vectorA[j] = prob
			for j,prob in topicModel[dictionary.doc2bow(sentencesB[k])]:
				vectorB[j] = prob
			if numpy.linalg.norm(vectorA) == 0 or numpy.linalg.norm(vectorB) == 0:
				features[k].append(0.)
			else:
				features[k].append(cosine_similarity(vectorA, vectorB))
		scores = model.predict(features)
		results = map(str, scores)
		with open(test_output[i], "wb") as f:
			f.write("\n".join(results))
