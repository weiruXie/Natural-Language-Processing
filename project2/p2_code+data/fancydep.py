import sys
import logging
from random import shuffle
from collections import defaultdict, Counter

logger = logging.getLogger("project")
logger.setLevel(logging.DEBUG)
f = logging.FileHandler("project.log", mode = "wb")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
f.setFormatter(formatter)
logger.addHandler(f)

class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    def dotProduct(self, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat]
        return dot

    def update(self, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y*val

class classifier(object):
    def __init__(self):
        self.weights = {"shift":Weights(),"left":Weights(),"right":Weights()}
        # self.updateTimes = 0.
        # self.tmp = {"shift":Weights(),"left":Weights(),"right":Weights()}
    def predict(self,feature):
        S = self.weights["shift"].dotProduct(feature)
        L = self.weights["left"].dotProduct(feature)
        R = self.weights["right"].dotProduct(feature)
        if S>L:
            if S>R:
                return "shift"
            else:
                return "right"
        else:
            if L>R:
                return "left"
            else:
                return "right"
    def update(self,feature,true_op, learning_rate = 1):
        P = self.predict(feature)
        if P!=true_op:
            self.weights[P].update(feature,-1*learning_rate)
            self.weights[true_op].update(feature,1*learning_rate)
            # self.updateTimes+=1
            # for move in self.weights:
            # 	self.tmp[move].update(self.weights[move],1)
        # else:
        #     self.weights[true_op].update(feature,0)

    # def finalize(self):
    # 	for move,weight in self.weights.iteritems():
    # 		for feat in weight:
    # 			weight[feat] = self.tmp[move][feat]/self.updateTimes


def read_dataset(subset):
	myFile = []
	sentence = []
	word = []
	if subset in [ 'en.dev', 'en.tr', 'en.tr100', 'en.tst']:
		with open(subset) as inp_hndl:
			for line in inp_hndl.readlines():
				items = line.split()
				if not items:
					myFile.append(sentence)
					sentence = []
				else:
					for index in range (0,len(items)):
						if index == 0 or index == 6:
							try:
								word.append(int(items[index]))
							except:
								word.append("_")
						else:
							word.append(items[index])
					sentence.append(word)
					word = []
		return myFile
	else:
		print '>>>> invalid input !!! <<<<<'

def testSentence(sentence,Classifier):
	sentence = [[0, "_", "_", "_","_", "_", "_"]] + sentence
	stack = [0]
	buff = []
	for i in range(1, len(sentence)):
		item = sentence[i]
		buff.append(item[0])
	buff.reverse()
	A = {}
	tailDict = defaultdict(list)
	while buff:
		tmp = stack[-1]
		a = sentence[tmp]
		b = sentence[buff[-1]]
		feats = getFeatures(stack, buff, a, b, sentence, tailDict)
		label = Classifier.predict(feats)

		logger.info("Stack:"+str(stack) + ",    Buffer:" + str(buff))
		logger.info("label:"+ str(label))

		if(label == "left"):
			if(a[0] == 0):
				transite(stack, buff, "shift")
			else:
				transite(stack, buff, "left")
				A[a[0]] = b[0]
				tailDict[b[0]].append(a[0])
 		elif(label == "shift"):
			transite(stack, buff, "shift")
		else:
			transite(stack, buff, "right")
			A[b[0]] = a[0]

	del sentence[0]
	return A

def write_file(outFile, file_name):
	with open(file_name, 'w') as outh:
		outh.write(outFile)

def test(myTestFile, clf, outFileName):
	newFile = []
	for sentence in myTestFile:
		A = testSentence(sentence, clf)
		newSent = []
		for word in sentence:
			word[6] = A.get(word[0], 0)
			newWord = "\t".join(map(str, word))
			newSent.append(newWord)
			print word
		newSent = "\n".join(newSent)
		newFile.append(newSent)
	newFile = "\n\n".join(newFile) + "\n\n\n\n\n"
	write_file(newFile,outFileName)


def trainSentence(sentence):
	sentence = [[0, "root", "_", "_","_", "_", "0"]] + sentence
	stack = [0]
	buff = []
	isHead = {}
	tailDict = defaultdict(list)
	for i in range(1, len(sentence)):
		item = sentence[i]
		buff.append(item[0])
		tailDict[item[6]].append(item[0])
		if item[6] in isHead:
			isHead[item[6]] += 1
		else:
			isHead[item[6]] = 1
	buff.reverse()

	while buff:
		a = sentence[stack[-1]]
		b = sentence[buff[-1]]
		feats = getFeatures(stack, buff, a, b, sentence, tailDict)
		label = ""
		
		if(a[6] == b[0]):
			transite(stack, buff, "left")
			label = "left"
			isHead[b[0]] -= 1
		elif(a[0] == b[6] and isHead.get(b[0], 0) == 0):
			transite(stack, buff, "right")
			label = "right"
			isHead[a[0]] -= 1
		else:
			transite(stack, buff, "shift")
			label = "shift"

		yield feats, label

def getFeatures(stack, buff, lWord, rWord,sentence, tailDict):
	feats = {}
	feats["stack:" + lWord[1]] = 1.
	feats["buffer:" + rWord[1]]= 1.
	feats["stackCPOS:" + lWord[3]] = 1.
	feats["bufferCPOS:" + rWord[3]] = 1.
	feats["stackFPOS:" + lWord[4]] = 1.
	feats["bufferFPOS:" + rWord[4]] = 1.
	feats["word_pair:" + lWord[1] + "_" + rWord[1]] = 1.
	feats["CPOS_pair:" + lWord[3]+ "_" + rWord[3]] = 1.
	feats["FPOS_pair:" + lWord[4]+ "_" + rWord[4]] = 1.

	#distance feature
	feats['dist:' + str(abs(lWord[0]-rWord[0]))] = 1.
	
	#identity, CPOS, FPOS of 2nd word in the stack and pair with top words 
	if(len(stack) > 1):
		stackLeft = sentence[stack[-2]]
		feats["stackLeftWord:" + stackLeft[1]] = 1.
		feats["stackLeftCPOS:" + stackLeft[3]] = 1.
		feats["stackLeftFPOS:" + stackLeft[4]] = 1.
		feats["LLWord_pair:" + stackLeft[1] + "_" + lWord[1]] = 1.
		feats["LRWord_pair:" + stackLeft[1] + "_" + rWord[1]] = 1.
		feats["LLCPOS_pair:" + stackLeft[3] + "_" + lWord[3]] = 1.
		feats["LRCPOS_pair:" + stackLeft[3] + "_" + rWord[3]] = 1.
		feats["LLFPOS_pair:" + stackLeft[4] + "_" + lWord[4]] = 1.
		feats["LRFPOS_pair:" + stackLeft[4] + "_" + rWord[4]] = 1.

	#stack[-2] and rWord
	if(len(stack) > 2):
		stackSecondLeft = sentence[stack[-3]]
		#feats["stackSecondLeftWord:" + stackSecondLeft[1]] = 1.
		#feats["LSLWord_pair:" + stackSecondLeft[1] + "_" + lWord[1]] = 1.
		feats["LSRWord_pair:" + stackSecondLeft[1] + "_" + rWord[1]] = 1.
		#feats["LSLPOS_pair:" + stackSecondLeft[3] + "_" + lWord[3]] = 1.
		feats["LSRCPOS_pair:" + stackSecondLeft[3] + "_" + rWord[3]] = 1.
		feats["LSRFPOS_pair:" + stackSecondLeft[4] + "_" + rWord[4]] = 1.

	#identity, CPOS, FPOS of 2nd word in the buff and pair with top words
	if(len(buff) > 1):
		buffRight = sentence[buff[-2]]
		feats["buffRightWord:" + buffRight[1]] = 1.
		feats["buffRightCPOS:" + buffRight[3]] = 1.
		feats["buffRightFPOS:" + buffRight[4]] = 1.
		feats["RLWord_pair:" + buffRight[1] + "_" + lWord[1]] = 1.
		feats["RRWord_pair:" + buffRight[1] + "_" + rWord[1]] = 1.
		feats["RLCPOS_pair:" + buffRight[3] + "_" + lWord[3]] = 1.
		feats["RLFPOS_pair:" + buffRight[4] + "_" + lWord[4]] = 1.
		feats["RRCPOS_pair:" + buffRight[3] + "_" + rWord[3]] = 1.
		feats["RRFPOS_pair:" + buffRight[4] + "_" + rWord[4]] = 1.

		# POS trigrams
		feats["Word_trigrams:" + lWord[1] + "_" + rWord[1] + "_" + buffRight[1]] = 1.
		feats["CPOS_trigrams:" + lWord[3] + "_" + rWord[3] + "_" + buffRight[3]] = 1.
		feats["FPOS_trigrams:" + lWord[4] + "_" + rWord[4] + "_" + buffRight[4]] = 1.

	if(len(buff) > 2):
		buffSecondRight = sentence[buff[-3]]
		feats["buffSecondRightWord:" + buffSecondRight[1]] = 1.
		feats["buffSecondRightCPOS:" + buffSecondRight[3]] = 1.
		feats["buffSecondRightFPOS:" + buffSecondRight[4]] = 1.
		#feats["RSLWord_pair:" + buffSecondRight[1] + "_" + lWord[1]] = 1.
		feats["RSRWord_pair:" + buffSecondRight[1] + "_" + rWord[1]] = 1.
		#feats["RSLPOS_pair:" + buffSecondRight[3] + "_" + lWord[3]] = 1.
		feats["RSRCPOS_pair:" + buffSecondRight[3] + "_" + rWord[3]] = 1.
		feats["RSRFPOS_pair:" + buffSecondRight[4] + "_" + rWord[4]] = 1.

	if(len(buff) > 3):
		buffSecondRight = sentence[buff[-3]]
		buffThirdRight = sentence[buff[-4]]
		# feats["buffThirdRightWord:" + buffThirdRight[1]] = 1.
		# feats["buffThirdRightCPOS:" + buffThirdRight[3]] = 1.
		# feats["buffThirdRightFPOS:" + buffThirdRight[4]] = 1.
		feats["3RSRWord_pair:" + buffThirdRight[1] + "_" + buffSecondRight[1]+"_"+rWord[1]] = 1.
		feats["3RSRCPOS_pair:" + buffThirdRight[3] + "_" + buffSecondRight[3]+"_"+rWord[3]] = 1.
		feats["3RSRFPOS_pair:" + buffThirdRight[4] + "_" + buffSecondRight[4]+"_"+rWord[4]] = 1.


	# valency
	if lWord[1] in tailDict:
		feats["S0_wv:" + lWord[1] + "_" + len(tailDict[lWord[1]])]
		feats["S0_Cpv:" + lWord[3] + "_" + len(tailDict[lWord[1]])]
	if rWord[1] in tailDict:
		feats["N0_wv:" + rWord[1] + "_" + len(tailDict[rWord[1]])]
		feats["N0_Cpv:" + rWord[3] + "_" + len(tailDict[rWord[1]])]

	return feats


def transite(stack, buff, label):
	if(label == "shift"):
		stack.append(buff.pop())
	elif(label == "left"):
		stack.pop()	
	else:
		buff.pop()
		buff.append(stack.pop())
		if not stack:
			transite(stack, buff, "shift")

# def generatePredict(sentence,Classifier):
#     stack = [0]
#     buff = []
#     for item in sentence:
#         buff.append(item[0])
#     buff.reverse()
#     while buff:
#         item1 = sentence[stack[-1]]
#         item2 = sentence[buff[-1]]
#         feats = getFeature(item1, item2)
#         transit(Classifier.predict(feats),stack,buff)


if __name__ == "__main__":
    clf = classifier()
    myTrainFile = read_dataset(sys.argv[1])
    
    trainingSet = []
    featureCount = Counter()
    for sentence in myTrainFile:
        for feature,label in trainSentence(sentence):
            #logger.info("feature: %s; label: %s"%(str(feature), str(label)))
            trainingSet.append([feature,label])
            for k in feature.keys():
            	featureCount[k] += 1

    print len(featureCount.keys())
    featureSet = set([k for k, v in featureCount.items() if v > 1])
    print len(featureSet)
    # for i in range(len(trainingSet)): 
    #     trainingSet[i][0] = {k: v for k, v in trainingSet[i][0].items() if k in featureSet}

    # clf.finalize()
    
    learning_rate = 1
    for i in range(10):
    	shuffle(trainingSet)
    	print i
    	for feat, label in trainingSet:
    		clf.update(feat, label, learning_rate)
    	learning_rate -= 0.1
    
    myTestFile = read_dataset(sys.argv[2])
    test(myTestFile, clf, sys.argv[3])
  #   for sentence in myTestFile:
		# for a, b in testSentence(sentence,oracal).iteritems():
		# 	print a,b


