import sys
from collections import deque
from random import shuffle

LEFT=1;RIGHT=2;SHIFT=3;Counter=1;corrHead=0;totalHead=0;LargeMargin=100

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


def featureAddWord(name, word, sentence, feats):
    if word == 0:
        feats[name + ':*root*'] = 1.
    else:
        feats[name + ':' + sentence.words[word-1]]= 1.

def featureAddWordWord(name, word1, word2, sentence, feats):
    if word1 == 0:
        w1 = '*root*'
    else:
        w1 = sentence.words[word1-1]
    
    if word2 == 0:
        w2 = '*root*'
    else:
        w2 = sentence.words[word2-1]
    
    feats[name + ':' + w1 + '_' + w2] = 1.

def featureAddCPos(name, cpos, sentence, feats):
    if cpos == 0:
        feats[name + ':*root*'] = 1.
    else:
        feats[name + ':' + sentence.coarsePOS[cpos-1]] = 1.

def featureAddCPosCPos(name, cpos1, cpos2, sentence, feats):
    if cpos1 == 0:
        c1 = '*root*'
    else:
        c1 = sentence.coarsePOS[cpos1-1]
    
    if cpos2 == 0:
        c2 = '*root*'
    else:
        c2 = sentence.coarsePOS[cpos2-1]
    
    feats[name + ':' + c1 + '_' + c2] = 1.

def featureAddFPos(name, fpos, sentence, feats):
    if fpos == 0:
        feats[name + ':*root*'] = 1.
    else:
        feats[name + ':' + sentence.finePOS[fpos-1]] = 1.

def featureAddFPosFPos(name, fpos1, fpos2, sentence, feats):
    if fpos1 == 0:
        f1 = '*root*'
    else:
        f1 = sentence.finePOS[fpos1-1]
    
    if fpos2 == 0:
        f2 = '*root*'
    else:
        f2 = sentence.finePOS[fpos2-1]
    
    feats[name + ':' + f1 + '_' + f2] = 1.

def featureAddWordCPos(name, word, pos, sentence, feats):
    if word == 0:
        w = '*root*'
    else:
        w = sentence.words[word-1]
    
    if pos == 0:
        p = '*root*'
    else:
        p = sentence.coarsePOS[pos-1]
    
    feats[name + ':' + w + '_' + p] = 1.

def featureAddWordFPos(name, word, pos, sentence, feats):
    if word == 0:
        w = '*root*'
    else:
        w = sentence.words[word-1]
    
    if pos == 0:
        p = '*root*'
    else:
        p = sentence.finePOS[pos-1]
    
    feats[name + ':' + w + '_' + p] = 1.

def featureAddCPos3(name, pos1, pos2, pos3, sentence, feats):
    if pos1 == 0:
        p1 = '*root*'
    else:
        p1 = sentence.coarsePOS[pos1-1]
    if pos2 == 0:
        p2 = '*root*'
    else:
        p2 = sentence.coarsePOS[pos2-1]
    if pos3 == 0:
        p3 = '*root*'
    else:
        p3 = sentence.coarsePOS[pos3-1]
    feats[name + ':' + p1 + '_' + p2 + '_' + p3] = 1.

def featureAddFPos3(name, pos1, pos2, pos3, sentence, feats):
    if pos1 == 0:
        p1 = '*root*'
    else:
        p1 = sentence.finePOS[pos1-1]
    if pos2 == 0:
        p2 = '*root*'
    else:
        p2 = sentence.finePOS[pos2-1]
    if pos3 == 0:
        p3 = '*root*'
    else:
        p3 = sentence.finePOS[pos3-1]
    feats[name + ':' + p1 + '_' + p2 + '_' + p3] = 1.


def featureAddGroup1(name, idx, sentence, feats):
    featureAddWord(name + 'w', idx, sentence, feats)
    featureAddCPos(name + 'c', idx, sentence, feats)
    featureAddFPos(name + 'f', idx, sentence, feats)

def featureAddGroup2(name, idx1, idx2, sentence, feats):
    featureAddWordWord(name + 'w', idx1, idx2, sentence, feats)
    featureAddCPosCPos(name + 'c', idx1, idx2, sentence, feats)
    featureAddFPosFPos(name + 'f', idx1, idx2, sentence, feats)


#represent configuration with features
def featureExtraction(stack, buffer, sentence):
    feats = {}
    
    if not stack or not buffer:
        return feats
    
    #identity of word at top of the stack
    #coarse POS of word at top of the stack
    #fine POS of word at top of the stack
    featureAddGroup1('0', stack[-1], sentence, feats)

    #identity of word at head of buffer
    #coarse POS of word at head of buffer
    #fine POS of word at head of buffer
    featureAddGroup1('1', buffer[0], sentence, feats)
    
    #pair of words at top of stack and head of buffer
    #pair of coarse POS at top of stack and head of buffer
    #pair of fine POS at top of stack and head of buffer
    featureAddGroup2('2', stack[-1], buffer[0], sentence, feats)
    
    #distance feature
    delta = stack[-1]
    beta = buffer[0]
    if delta != 0 and beta != 0:
        feats['dist:' + str(abs(delta-beta))] = 1.
    
    #identity of 2nd word in the buffer
    #coarse POS of 2nd word in buffer
    #fine POS of 2nd word in buffer
    if len(buffer) > 1: featureAddGroup1('3', buffer[1], sentence, feats)
    
    #word pair in buffer
    #coarse POS pair in buffer
    #fine POS pair in buffer
    if len(buffer) > 1: featureAddGroup2('4', buffer[0], buffer[1], sentence, feats)
    
    #skip word pair of stack and 2nd word in buffer
    #skip coarse POS pair of stack and 2nd word in buffer
    #skip fine POS pair of stack and 2nd word in buffer
    if len(buffer) > 1: featureAddGroup2('5', stack[-1], buffer[1], sentence, feats)
    
    #word pair in stack
    #coarse POS pair in stack
    #fine POS pair in stack
    if len(stack) > 1: featureAddGroup2('6', stack[-2], stack[-1], sentence, feats)
    
    #S0wp
    featureAddWordFPos('7', stack[-1], stack[-1], sentence, feats)
    
    #N2p
    if len(buffer) > 2:
        featureAddGroup1('8', buffer[2], sentence, feats)
        featureAddGroup2('9', buffer[0], buffer[2], sentence, feats)
    
    #stack[-2] - buffer[0]
    if len(stack) > 1: featureAddGroup2('10', stack[-2], buffer[0], sentence, feats)
    
    # POS trigrams
    if len(buffer) > 1:
        featureAddCPos3('11', stack[-1], buffer[0], buffer[1], sentence, feats)
        featureAddFPos3('12', stack[-1], buffer[0], buffer[1], sentence, feats)
    
    # context of buffer[0] in sentence
    if buffer[0] > 1:
        featureAddGroup1('13', buffer[0]-1, sentence, feats)
        featureAddGroup2('14', buffer[0]-1, buffer[0], sentence, feats)
    
    # context of stack[-1] in sentence
    if stack[-1] < len(sentence.words):
        featureAddGroup1('15', stack[-1]+1, sentence, feats)
        featureAddGroup2('16', stack[-1], stack[-1]+1, sentence, feats)
    
    
    return feats


def getPredictTransition(feats):
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

def getPredictTransitionLargeMargin(feats, trueT):
    global perceptron,LEFT,RIGHT,SHIFT,LargeMargin
    lw = perceptron.lWeights.dotProduct(feats) + LargeMargin
    rw = perceptron.rWeights.dotProduct(feats) + LargeMargin
    sw = perceptron.sWeights.dotProduct(feats) + LargeMargin
    
    if trueT == LEFT:
        lw -= LargeMargin
    elif trueT == RIGHT:
        rw -= LargeMargin
    else:
        sw -= LargeMargin

    if lw > rw and lw > sw:
        return LEFT
    elif rw > sw:
        return RIGHT
    else:
        return SHIFT

def getAvgPredictTransition(feats):
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

def getAvgPredictTransitionII(stack, buffer, sentence, step):
    global LEFT,RIGHT,SHIFT
    #try LEFT,RIGHT,SHIFT action
    scoreLeft = search(LEFT, stack, buffer, sentence, step-1)
    scoreRight = search(RIGHT, stack, buffer, sentence, step-1)
    scoreShift = search(SHIFT, stack, buffer, sentence, step-1)

    if scoreLeft > scoreRight and scoreLeft > scoreShift:
	return LEFT
    elif scoreRight > scoreShift:
	return RIGHT
    else:
	return SHIFT

def getLeftScores(feats):
    global perceptron,averagedPerceptron
    return perceptron.lWeights.dotProduct(feats) - averagedPerceptron.lWeights.dotProduct(feats)/Counter 

def getRightScores(feats):
    global perceptron,averagedPerceptron
    return perceptron.rWeights.dotProduct(feats) - averagedPerceptron.rWeights.dotProduct(feats)/Counter

def getShiftScores(feats):
    global perceptron,averagedPerceptron
    return perceptron.sWeights.dotProduct(feats) - averagedPerceptron.sWeights.dotProduct(feats)/Counter

def updateLeft(stack, buffer):
    cur = stack.pop()
    return cur

def updateRight(stack, buffer):
    cur = stack.pop()
    bur = buffer.popleft()
    buffer.appendleft(cur)
    return bur

def updateShift(stack, buffer):
    stack.append(buffer.popleft())
def restoreLeft(stack, buffer, val):
    stack.append(val)

def restoreRight(stack, buffer, val):
    stack.append(buffer.popleft())
    buffer.appendleft(val)

def restoreShift(stack, buffer):
    buffer.appendleft(stack.pop())
    

def search(action, stack, buffer, sentence, step):
    global LEFT,RIGHT,SHIFT
    if len(stack) == 0 or len(buffer) == 0:
	return 0
    feats = featureExtraction(stack, buffer, sentence)
    if step == 0:
	if action == LEFT:
	    return getLeftScores(feats)
	elif action == RIGHT:
	    return getRightScores(feats)
	else:
	    return getShiftScores(feats)

    if action == LEFT:
	scoreMax = getLeftScores(feats)
	leftRes = updateLeft(stack, buffer)
    elif action == RIGHT:
	scoreMax = getRightScores(feats)
	rightRes = updateRight(stack, buffer)
    else:
	scoreMax = getShiftScores(feats)
	updateShift(stack, buffer)

    scoreLeft = search(LEFT, stack, buffer, sentence, step-1)
    scoreRight = search(RIGHT, stack, buffer, sentence, step-1)
    scoreShift = search(SHIFT, stack, buffer, sentence, step-1)

    if scoreLeft > scoreRight and scoreLeft > scoreShift:
	scoreMax += scoreLeft
    elif scoreRight > scoreShift:
	scoreMax += scoreRight
    else:
	scoreMax += scoreShift

    if action == LEFT:
	restoreLeft(stack, buffer, leftRes)
    elif action == RIGHT:
	restoreRight(stack, buffer, rightRes)
    else:
	restoreShift(stack, buffer)

    return scoreMax

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
    sentences = []
    sentence = Sentence()
    trainErr = 0.
    totalt = 0.

    # read all sentences and store them
    for line in open(sys.argv[1], 'r'):
        line = line.strip()
        if line == "":
            sentences.append(sentence)
            sentence = Sentence()
        else:
            sentence.update(line,True)

    #shuffle the sentences before training
    shuffle(sentences)
    for sentence in sentences:
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
                t2 = getPredictTransitionLargeMargin(feats,t1)
		if t1 != t2:
		    trainErr += 1
                updateWeights(t1, t2,feats)
	    updateConfiguration(stack, buffer, sentence, t1)
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
                    #t2 = getAvgPredictTransition(feats)
		    step = 1
                    t2 = getAvgPredictTransitionII(stack, buffer, sentence, step)
                    if t1 != t2:
                        validationErr += 1
                updateConfiguration(stack, buffer, sentence, t1)
            sentence = Sentence()
        else:
            sentence.update(line,True)
    return validationErr/totalv

def predict():
    global SHIFT,corrHead,totalHead
    sentence = Sentence()
    lines = []
    corrHead = 0
    totalHead = 0
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
                    #t = getAvgPredictTransition(feats)
		    #t = getOracleTransition(stack, buffer, sentence)
		    step = 1
                    t = getAvgPredictTransitionII(stack, buffer, sentence, step)
		    #update head
		    sentence.updateRelation(t, stack, buffer)		    
		else:
		    t = SHIFT
		updateConfigurationII(stack, buffer, t)
	    ###write updated sentence to output file
	    for curLine in lines:
		fields = curLine.split('\t')
		trueHead = fields[6]
		if int(fields[0]) in sentence.head:
		    fields[6] = str(sentence.head[int(fields[0])])
		else:
		    fields[6] = '0'
		if trueHead == fields[6]:
		    corrHead += 1
		totalHead += 1
		output.write('\t'.join(fields))
		output.write('\n')
	    output.write('\n')
	    lines = []
            sentence = Sentence()
        else:
            sentence.update(line,False)
	    lines.append(line)


for iteration in range(20):
    print str(iteration) + " : " + str(train()) + ", " + str(validate())
    predict()
    print "%f%% (%d/%d)" % (float(corrHead)/totalHead*100, corrHead, totalHead)
