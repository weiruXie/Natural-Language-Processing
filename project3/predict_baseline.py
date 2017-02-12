import sys
import nltk
import numpy
from scipy import spatial
from nltk.corpus import stopwords

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

	stopWords = stopwords.words('english')
	ascistop =[]
	for word in stopWords:
		word = word.encode("utf8")
		ascistop.append(word)
	stopWordsSet = set(ascistop[1:60])

	sentences_A = map(lambda sent: filter(lambda word: not word in stopWordsSet, sent), sentences_A)
	sentences_B = map(lambda sent: filter(lambda word: not word in stopWordsSet, sent), sentences_B)
	return (sentences_A, sentences_B)

def cosine_similarity(listA, listB):
	result = 1 - spatial.distance.cosine(listA, listB)
	return result

def sentence_vector_similarity(A, B, embeddings, output):
	dimensions = 300

	f_in = open(output, "wb")
	for i in range(len(A)):
		vectorA = numpy.zeros(dimensions)
		vectorB = numpy.zeros(dimensions)
		for wordA in A[i]:
			if wordA in embeddings:
				vectorA += embeddings[wordA]
		for wordB in B[i]:
			if wordB in embeddings:
				vectorB += embeddings[wordB]
		if numpy.linalg.norm(vectorA) == 0 or numpy.linalg.norm(vectorB) == 0:
			cos_s = 0.
		else:
			cos_s = cosine_similarity(vectorA, vectorB)
		f_in.write(str(cos_s) + "\n")
	f_in.close()

if __name__ == "__main__":
	embeddings = {}
	print "start reading"
	with open('paragram-phrase-XXL.txt') as pretrained:
		for line in pretrained:
			wordL = line.strip().split()
			for i in xrange(1,len(wordL)):
				wordL[i] = float(wordL[i])
			embeddings[wordL[0]] = numpy.array(wordL[1:])
	print "size: %d"%len(embeddings)
	with open('paragram_300_sl999.txt') as pretrained:
		for line in pretrained:
			wordL = line.strip().split()
			for i in xrange(1,len(wordL)):
				wordL[i] = float(wordL[i])
			if(wordL[0] not in embeddings):
				embeddings[wordL[0]] = numpy.array(wordL[1:])
	print "size: %d"%len(embeddings)
	print "finish reading"
	sentencesA, sentencesB = read_sentences(sys.argv[1])
	sentence_vector_similarity(sentencesA, sentencesB, embeddings, sys.argv[2])
