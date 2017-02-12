"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""

"""
 read one of train, dev, test subsets 
 
 subset - one of train, dev, test
 
 output is a tuple of three lists
 	labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
 	targets: the index within the text of the token to be disambiguated
 	texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk 
def read_dataset(subset):
	labels = []
	texts = []
	targets = []
	if subset in ['train', 'dev', 'test', 'mylabels']:
		with open('data/wsd_'+subset+'.txt') as inp_hndl:
			for example in inp_hndl:
				label, text = example.strip().split('\t')   # strip() is to remove space at the begining and tail
				text = nltk.word_tokenize(text.lower().replace('" ','"'))  # make "I want a job" to be ["I", "want", "a", "job"]
				if 'line' in text:
					ambig_ix = text.index('line')
				elif 'lines' in text:
					ambig_ix = text.index('lines')
				else:
					ldjal
				targets.append(ambig_ix)
				labels.append(label)
				texts.append(text)
		return (labels, targets, texts)
	else:
		print '>>>> invalid input !!! <<<<<'

"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics
def eval(gold_labels, predicted_labels):
	return ( sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
			 sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro') )


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""
def write_predictions(predictions, file_name):
	with open(file_name, 'w') as outh:
		for p in predictions:
			outh.write(p+'\n')

"""
predicte the class given a sentence
"""
def predict(X, classes, p_s, textClass):
	dictP_X_s = {}
	for i in s:
		prior = 1.0
		for word in X:
			prior *= float(textClass[i].count(word) + 1) / float (len(textClass[i]) + V)
			dictP_X_s['X'+'|'+ i] = prior
	maxP = float('-inf')
	predict = "" 
	dictP_s_X = {}
	for i in s:
		temp = dictP_X_s['X' + '|' + i] * p_s[i]
		dictP_s_X[i + '|' + 'X'] = temp
		if temp > maxP:
			maxP = temp
			predict = i
	return predict

"""
predict class, with new feature of higher weight for words appearing before and after 'line' or 'lines'
"""
def predic_before_and_after(X, classs, p_s, wordsBefore, wordsAfter, textClass):
	dictP_X_s = {}
	for i in s:
		prior = 1.0	
		for word in X:
			prior *= float(textClass[i].count(word) + 1) / float (len(textClass[i]) + V)
		print i, prior
		indexI = 0
		if 'line' in X:
			indexI = X.index('line')
		elif 'lines' in X:
			indexI = X.index('lines')

		if X[indexI-1] in wordsBefore:
			if indexI == 0:
				prior *= float(wordsBefore['*']) / float(sum(wordsBefore.values()))
			else:
				prior *= float(wordsBefore[X[indexI-1]]) / float(sum(wordsBefore.values()))

		if X[indexI+1] in wordsAfter:
			if indexI == len(X):
				prior *= float(wordsAfter['*']) / float(sum(wordsAfter.values()))	
			else:
				prior *= float(wordsAfter[X[indexI+1]]) / float(sum(wordsAfter.values()))

		# print prior
		# assert 0

		# if indexI == 0:
		# 	prior *= float(wordsBefore['*']) / float(sum(wordsBefore.values()))
		# elif indexI == len(X):
		# 	prior *= float(wordsAfter['*']) / float(sum(wordsAfter.values()))
		# else:
		# 	if X[indexI-1] in wordsBefore:
		# 		prior *= float(wordsBefore[X[indexI-1]]) / float(sum(wordsBefore.values()))
		# 	if X[indexI+1] in wordsAfter:
		# 		prior *= float(wordsAfter[X[indexI+1]]) / float(sum(wordsAfter.values()))

		dictP_X_s['X'+'|'+ i] = prior

	maxP = float('-inf')
	predict = ""
	dictP_s_X = {}
	for i in s:
		temp = dictP_X_s['X' + '|' + i] * p_s[i]
		dictP_s_X[i + '|' + 'X'] = temp
		print i, temp
		if temp > maxP:
			maxP = temp
			predict = i
	print predict
	return predict


"""
remove lowest and hignest frequent words
"""
def preprocess(train_texts, train_targets,train_labels,x):

    processed_train_texts = []
    processed_train_targets = []
    processed_train_labels = train_labels
    word_frequency = {}
    n = len(train_texts)
    for i in xrange(n):
        sentenct = train_texts[i]
        for word in sentenct:
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
    high = 0
    for item in word_frequency:
        if word_frequency[item] > high:
            high = word_frequency[item]
    high = high - x
    low = x
    for i in xrange(n):
        sentenct = train_texts[i]
        temp = []
        index = 0
        for word in sentenct:
            if word_frequency[word] <= high and word_frequency > low:
                temp.append(word)
        if 'line' in temp:
            index = temp.index('line')
        elif 'lines' in temp:
            index = temp.index( 'lines')
        processed_train_targets.append(index)
        processed_train_texts.append(temp)
    #print processed_train_texts[:3]
    return (processed_train_texts, processed_train_targets, processed_train_labels)
pass

"""
preprocess 2: replace words with their roots
"""
from nltk.stem.lancaster import LancasterStemmer
def preprocess2(train_texts, train_targets,train_labels,test_texts, test_targets, test_labels):
    lancaster_stemmer = LancasterStemmer()
    n = len(train_texts)
    for i in xrange(n):
        sentenct = train_texts[i]
        for j in xrange(len(sentenct)):
            sentenct[j] = lancaster_stemmer.stem(sentenct[j])
    n = len(test_texts)
    for i in xrange(n):
        sentenct = test_texts[i]
        for j in xrange(len(sentenct)):
            sentenct[j] = lancaster_stemmer.stem(sentenct[j])
	pass


def getAccuracy(gold_labels, predicted_labels):
	count = 0
	length = len(gold_labels)
	for i in range (0, length):
		if gold_labels[i] == predicted_labels[i]:
			count += 1
	accuracy = float(count) / float(length)
	return accuracy

def Cohen_Kappa(gold_labels, predicted_labels, s):
	length = len(gold_labels)
	arr = [[0]*6 for _ in xrange(6)]
	for i in range (0, length):
		x = s.index(gold_labels[i])
		y = s.index(predicted_labels[i])
		arr[x][y] += 1
	a = 0
	ef = 0
	for i in range (0, 6):
		a += arr[i][i]
		ef += sum(arr[i]) * sum(list(zip(*arr)[i]))
	ef = ef / length
	K = float (a - ef) / float (length - ef)
	return K

def get_class_text (train_texts, train_labels):
	tt = {}
	for i in range(0, len(train_labels)):
		if train_labels[i] in tt:
			tt[train_labels[i]] += train_texts[i]
		else:
			tt[train_labels[i]] = train_texts[i]
	return tt


"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
from collections import Counter
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):	
	predicts = []
	s = ['cord', 'division', 'formation', 'phone', 'product', 'text']
	l = len(train_labels)
	p_s = Counter(train_labels)
	for key in p_s.keys():
		p_s[key] = float (p_s[key]) / float (l)
	for i in range (0, len(test_texts)):
		p = predict(test_texts[i], s, p_s, textOfClass)
		predicts.append(p)
	write_predictions(predicts, 'q4p2.txt')
	accuracy = getAccuracy(test_labels, predicts)
	print accuracy
	return eval(test_labels, predicts)
	pass

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final classifier implementation of part 3 goes here**
	"""
	pass


"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
# def run_extended_bow_naivebayes_classifier_FeatureA(train_texts, train_targets,train_labels, 
# 				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
# 	preprocess2(train_texts, train_targets,train_labels,test_texts, test_targets, test_labels)
# 	predicts = []
# 	s = ['cord', 'division', 'formation', 'phone', 'product', 'text']
# 	l = len(train_labels)
# 	p_s = Counter(train_labels)
# 	for key in p_s.keys():
# 		p_s[key] = float (p_s[key]) / float (l)
# 	text_each_Class = get_class_text(train_texts, train_labels)
# 	for i in range (0, len(test_texts)):
# 		p = predict(test_texts[i], s, p_s, text_each_Class)
# 		predicts.append(p)
# 	write_predictions(predicts, 'q4p4_nb_featureA.txt')
# 	accuracy = getAccuracy(test_labels, predicts)
# 	print accuracy
# 	return eval(test_labels, predicts)
# 	pass


def predictWithNewFeatures(sentence, wordsBefore, wordsAfter, wordCount_class, classPrior, targetIndex, minP):
	maxProb = -1
	predict = None
	for label in classPrior:
		prob = classPrior.get(label)
		for word in sentence:
			prob *= wordCount_class.get(label).get(word, minP)

		if targetIndex < 1:
			prob *= wordsBefore.get(label).get('*', minP)
		else:
			prob *= wordsBefore.get(label).get(sentence[targetIndex-1], minP)

		if targetIndex > len(sentence)-1:
			prob *= wordsAfter.get(label).get('*', minP)
		else:
			prob *= wordsAfter.get(label).get(sentence[targetIndex+1], minP)

		if prob > maxProb:
			maxProb = prob
			predict = label

	return predict

from collections import defaultdict
def run_extended_bow_naivebayes_classifier_Feature(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	preprocess2(train_texts, train_targets,train_labels,test_texts, test_targets, test_labels)

	wordsBefore = defaultdict(Counter)
	wordsAfter = defaultdict(Counter)
	for j, index in enumerate(train_targets):
		tokens = train_texts[j]
		label = train_labels[j]
		try:
			wordsBefore[label][tokens[index-1]] += 1
		except:
			wordsBefore[label]['*'] += 1

		try:
			wordsAfter[label][tokens[index+1]] += 1
		except:
			wordsAfter[label]['*'] += 1


	predicts = []
	s = ['cord', 'division', 'formation', 'phone', 'product', 'text']
	l = len(train_labels)
	p_s = Counter(train_labels)
	for key in p_s.keys():
		p_s[key] = float (p_s[key]) / float (l)

	# print "start estimating conditional prob"
	# count the words for each class
	wordCount_class = defaultdict(Counter)
	for tokens, label in zip(train_texts, train_labels):
		wordCount_class[label] += Counter(tokens)

	# normalize
	for label, count in wordCount_class.items():
		totalCount = float(sum(count.values()))
		for word, value in count.items():
			wordCount_class[label][word] /= totalCount

	# textForClass = get_class_text(train_texts, train_labels)
	minP = min(map(lambda x: min(x.values()), wordCount_class.values())) / 2

	# print "start predicting"
	for i in range (0, len(test_texts)):
		p = predictWithNewFeatures(test_texts[i], wordsBefore, wordsAfter, wordCount_class, p_s, test_targets[i], minP)
		# p = predic_before_and_after(test_texts[i], s, p_s, wordsBefore, wordsAfter, textForClass)
		# print p
		predicts.append(p)
	write_predictions(predicts, 'q4p4_nb.txt')
	accuracy = getAccuracy(test_labels, predicts)
	print accuracy
	return eval(test_labels, predicts)



"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass


from collections import Counter
import operator
if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')
    mylabels_labels, mylabels_targets, mylabels_texts = read_dataset('mylabels')
    processed_train_texts, processed_train_targets, processed_train_labels = preprocess(train_texts, train_targets,train_labels,2)
    
    # Arrays for classes and words
    s = ['cord', 'division', 'formation', 'phone', 'product', 'text']
    w = ['time', 'loss', 'export']

    # Calculate the baseline(part1_1) and count c_s(part2_1)
    c_s = Counter(train_labels)
    c_s_dev = Counter(dev_labels)
    most_frequent_class_train = max(c_s.iteritems(), key=operator.itemgetter(1))[0]
    baseline_train = float (c_s.get(most_frequent_class_train)) / len(train_labels) 
    most_frequent_class_dev = max(c_s_dev.iteritems(), key=operator.itemgetter(1))[0]
    baseline_dev = float (c_s_dev.get(most_frequent_class_dev)) / len(dev_labels)

	# Use Cohen's Kappa to calculate the inner agreement   
    K = Cohen_Kappa(dev_labels[:20], mylabels_labels, s)

    # count c(s,w)
    c_sw = {}
    for i in range(0, len(train_labels)):
    	for word in w:
    		key = train_labels[i]+','+word
    		if key in c_sw:
    			temp = c_sw.get(key) 
    			temp += train_texts[i].count(word)
    			c_sw[key] = temp
    		else:
    			c_sw[key] = train_texts[i].count(word)

    # Calculate p(s)
    l = len(train_labels)
    p_s = c_s.copy()
    for key in p_s.keys():
    	p_s[key] = float (p_s[key]) / float (l)

    # Calculate p(w|s)
    word_set = set([]) 
    for sentence in train_texts:
    	for word in sentence:
    		word_set.add(word)
    V = len(word_set)
    textOfClass = get_class_text(train_texts, train_labels)
    p_ws = {}
    for i in s:
    	for j in w: 
    		temp = float(c_sw[i+','+j] + 1) / float (len(textOfClass[i]) + V)
    		p_ws[j+'|'+i] = temp
    
    # Calculate P(s|X)
    X = "and i can tell you that i 'm an absolute nervous wreck every time she performs . i have her practice the last two lines on each page , so I can learn exactly when to turn the page -- just one of the tricks to this trade that i 've learned the hard way ."
    X = nltk.word_tokenize(X.lower().replace('" ','"'))
    p = predict(X, s, p_s, textOfClass)

    # running the Bayes classifier
    test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)
    print test_scores

    # Bayes_Feature B
    test_scores = run_extended_bow_naivebayes_classifier_Feature(processed_train_texts, processed_train_targets, processed_train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
    print test_scores

    

