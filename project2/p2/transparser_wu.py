#!/usr/bin/python
import sys
from collections import deque

LEFT=1;RIGHT=2;SHIFT=3;Counter=1

class Sentence():
    def __init__(self):
	self.head = {}
	self.sonCount = {}
	self.words = []
	self.finePOS = []
	self.coarsePOS = []
	
    def update(self,line,flag):
	self.cur = line.split('\t')
	self.words.append(self.cur[1])
	self.finePOS.append(self.cur[4])
	self.coarsePOS.append(self.cur[3])
	if flag:
	    self.head[int(self.cur[0])] = int(self.cur[6])
	    self.sonCount[self.cur[6]] = self.sonCount.get(self.cur[6], 0) + 1

    def updateRelation(self, t, stack, buffer):
	global LEFT,RIGHT,SHIFT
	delta = stack[-1]
	beta = buffer[0]
	if t == LEFT:
	    self.head[delta] = beta
	elif t == RIGHT:
	    self.head[beta] = delta
	return
	    

class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    # given a feature vector, compute a dot product
    def dotProduct(self, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat]
        return dot

    # given an example _and_ a true label (y is +1 or -1), update the
    # weights according to the perceptron update rule (we assume
    # you've already checked that the classification is incorrect
    def update(self, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y * val

#perceptron
class Model():
    def __init__(self):
	self.lWeights = Weights()
	self.rWeights = Weights()
	self.sWeights = Weights()
perceptron = Model()
averagedPerceptron = Model()

#arc-standard transition system
def getOracleTransition(stack, buffer, sentence):
    global LEFT,RIGHT,SHIFT

    #shift if stack is empty
    if not stack:
	return SHIFT
    delta = stack[-1]
    beta = buffer[0]
    if delta in sentence.head and sentence.head[delta] == beta:
	return LEFT
    elif beta in sentence.head and sentence.head[beta] == delta and (str(beta) not in sentence.sonCount or sentence.sonCount.get(str(beta)) == 0):
	return RIGHT
    else:
	return SHIFT

#update configuration	
def updateConfiguration(stack, buffer, sentence, action):
    if action == LEFT:
	stack.pop()
	sentence.sonCount[str(buffer[0])] -= 1;
    elif action == RIGHT:
	cur = stack.pop()
	sentence.sonCount[str(cur)] -= 1
	buffer.popleft()
	buffer.appendleft(cur)
    else:
	stack.append(buffer.popleft())
    return

#update configurationII
def updateConfigurationII(stack, buffer, action):
    if action == LEFT:
        stack.pop()
    elif action == RIGHT:
        cur = stack.pop()
        buffer.popleft()
        buffer.appendleft(cur)
    else:
        stack.append(buffer.popleft())
    return


#represent configuration with features
def featureExtraction(stack, buffer, sentence):
    feats = {}
    #identity of word at top of the stack
    delta = stack[-1]
    if delta == 0:
        wordS = '*root*'
    else:
	wordS = sentence.words[delta-1]
    feats['stop:' + wordS] = 1.
    
    #coarse POS of word at top of the stack
    if delta != 0:
        cposS = sentence.coarsePOS[delta-1]
	feats['stopcpos:' + cposS] = 1.
	
    #identity of word at head of buffer
    beta = buffer[0]
    if beta == 0:
	wordB = '*root*'
    else:
	wordB = sentence.words[beta-1]
    feats['btop:' + wordB] = 1.

    #coarse POS of word at head of buffer
    if beta != 0:
        cposB = sentence.coarsePOS[beta-1]
	feats['btopcpos:' + cposB] = 1.

    #pair of words at top of stack and head of buffer
    feats['word_pair:' + wordS + '_' + wordB] = 1.

    #pair of coarse POS at top of stack and head of buffer
    if delta != 0 and beta != 0:
        feats['cpos_pair:' + cposS + '_' + cposB] = 1.
    return feats

def getPredictTransition(stack, buffer, sentence,feats):
    global perceptron,LEFT,RIGHT,SHIFT
    lw = perceptron.lWeights.dotProduct(feats) 
    rw = perceptron.rWeights.dotProduct(feats)
    sw = perceptron.sWeights.dotProduct(feats)
    if lw > rw and lw > sw:
	return LEFT
    elif rw > sw:
	return RIGHT
    else:
	return SHIFT

def getAvgPredictTransition(stack, buffer, sentence,feats):
    global perceptron,averagedPerceptron,LEFT,RIGHT,SHIFT,Counter
    lw = perceptron.lWeights.dotProduct(feats) - averagedPerceptron.lWeights.dotProduct(feats)/Counter
    rw = perceptron.rWeights.dotProduct(feats) - averagedPerceptron.rWeights.dotProduct(feats)/Counter
    sw = perceptron.sWeights.dotProduct(feats) - averagedPerceptron.sWeights.dotProduct(feats)/Counter
    if lw > rw and lw > sw:
        return LEFT
    elif rw > sw:
        return RIGHT
    else:
        return SHIFT


def updateWeights(trueT,predT,feats):
    global perceptron,averagedPerceptron,LEFT,RIGHT,SHIFT,Counter
    if trueT != predT:
	if trueT == LEFT:
	    perceptron.lWeights.update(feats, 1)
	    averagedPerceptron.lWeights.update(feats, Counter)
	elif trueT == RIGHT:
	    perceptron.rWeights.update(feats, 1)
	    averagedPerceptron.rWeights.update(feats, Counter)
	else:
	    perceptron.sWeights.update(feats, 1)
	    averagedPerceptron.sWeights.update(feats, Counter)
	if predT == LEFT:
	    perceptron.lWeights.update(feats, -1)
	    averagedPerceptron.lWeights.update(feats, -Counter)
	elif predT == RIGHT:
	    perceptron.rWeights.update(feats, -1)
            averagedPerceptron.rWeights.update(feats, -Counter)
	else:
	    perceptron.sWeights.update(feats, -1)
	    averagedPerceptron.sWeights.update(feats, -Counter)
    Counter += 1


def train():
    sentence = Sentence()
    trainErr = 0.
    totalt = 0.
    for line in open(sys.argv[1], 'r'):
        line = line.strip()
        if line == "":
            #initialize Stack, Buffer
	    stack = [0]
	    buffer = deque([])
	    for i in range(1, len(sentence.words)+1):
	        buffer.append(i)
	    while len(buffer) != 0:
                t1 = getOracleTransition(stack, buffer, sentence)
		totalt += 1
		if len(stack) != 0 and len(buffer) != 0:
		    feats = featureExtraction(stack, buffer,sentence)
                    t2 = getPredictTransition(stack, buffer, sentence,feats)
		    if t1 != t2:
			trainErr += 1
                    updateWeights(t1, t2,feats)
		updateConfiguration(stack, buffer, sentence, t1)
	    sentence = Sentence()
	else:
	    sentence.update(line,True)
    return trainErr/totalt

def validate():
    validationErr = 0.
    totalv = 0.
    sentence = Sentence()
    for line in open(sys.argv[2], 'r'):
        line = line.strip()
        if line == "":
            #initialize Stack, Buffer
            stack = [0]
            buffer = deque([])
            for i in range(1, len(sentence.words)+1):
                buffer.append(i)
            while len(buffer) != 0:
                t1 = getOracleTransition(stack, buffer, sentence)
		totalv += 1
                if len(stack) != 0 and len(buffer) != 0:
                    feats = featureExtraction(stack, buffer,sentence)
                    t2 = getAvgPredictTransition(stack, buffer, sentence,feats)
                    if t1 != t2:
                        validationErr += 1
                updateConfiguration(stack, buffer, sentence, t1)
            sentence = Sentence()
        else:
            sentence.update(line,True)
    return validationErr/totalv

def predict():
    global SHIFT
    sentence = Sentence()
    lines = []
    output = open(sys.argv[3], 'w')
    for line in open(sys.argv[2], 'r'):
        line = line.strip()
        if line == "":
            #initialize Stack, Buffer
            stack = [0]
            buffer = deque([])
            for i in range(1, len(sentence.words)+1):
                buffer.append(i)
            while len(buffer) != 0:
                if len(stack) != 0 and len(buffer) != 0:
                    feats = featureExtraction(stack, buffer,sentence)
                    t = getAvgPredictTransition(stack, buffer, sentence,feats)
		    #update head
		    sentence.updateRelation(t, stack, buffer)		    
		else:
		    t = SHIFT
		updateConfigurationII(stack, buffer, t)
	    ###write updated sentence to output file
	    for curLine in lines:
		fields = curLine.split('\t')
		if int(fields[0]) in sentence.head:
		    fields[6] = str(sentence.head[int(fields[0])])
		else:
		    fields[6] = '0'
		output.write('\t'.join(fields))
		output.write('\n')
	    output.write('\n')
	    lines = []
            sentence = Sentence()
        else:
            sentence.update(line,False)
	    lines.append(line)


for iteration in range(30):
    print str(iteration) + " : " + str(train())

predict()
