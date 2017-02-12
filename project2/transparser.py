import sys
import logging
from random import shuffle

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
        self.updateTimes = 0.
        self.tmp = {"shift":Weights(),"left":Weights(),"right":Weights()}
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
    def update(self,feature,true_op):
        P = self.predict(feature)
        if P!=true_op:
            self.weights[P].update(feature,-1)
            self.weights[true_op].update(feature,1)
            self.updateTimes+=1
            for move in self.weights:
            	self.tmp[move].update(self.weights[move],1)
        # else:
        #     self.weights[true_op].update(feature,0)
    def finalize(self):
    	for move,weight in self.weights.iteritems():
    		for feat in weight:
    			weight[feat] = self.tmp[move][feat]/self.updateTimes


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
	# print stack,
	A = {}
	while buff:
		tmp = stack[-1]
		a = sentence[tmp]
		b = sentence[buff[-1]]
		feats = getFeatures(a, b)
		label = Classifier.predict(feats)

		logger.info("Stack:"+str(stack) + ",    Buffer:" + str(buff))
		logger.info("label:"+ str(label))

		if(label == "left"):
			if(a[0] == 0):
				transite(stack, buff, "shift")
			else:
				transite(stack, buff, "left")
				A[a[0]] = b[0]
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
		newSent = "\n".join(newSent)
		newFile.append(newSent)
	newFile = "\n\n".join(newFile) + "\n\n\n\n\n"
	write_file(newFile, outFileName)


def trainSentence(sentence):
	sentence = [[0, "root", "_", "_","_", "_", "0"]] + sentence
	stack = [0]
	buff = []
	isHead = {}
	for i in range(1, len(sentence)):
		item = sentence[i]
		buff.append(item[0])
		if item[6] in isHead:
			isHead[item[6]] += 1
		else:
			isHead[item[6]] = 1
	buff.reverse()

	while buff:
		a = sentence[stack[-1]]
		b = sentence[buff[-1]]
		feats = getFeatures(a, b)
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

def getFeatures(lWord, rWord):
	feats = {}
	feats["stack:" + lWord[1]] = 1.
	feats["buffer:" + rWord[1]]= 1.
	feats["stackPOS:" + lWord[3]] = 1.
	feats["bufferPOS:" + rWord[3]] = 1.
	feats["word_pair:" + lWord[1] + "_" + rWord[1]] = 1.
	feats["POS_pair:" + lWord[3]+ "_" + rWord[3]] = 1.
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
    for sentence in myTrainFile:
        for feature,label in trainSentence(sentence):
            trainingSet.append((feature,label))

    clf.finalize()
    
    for i in range(10):
    	# shuffle(trainingSet)
    	print i
    	for feat, label in trainingSet:
    		clf.update(feat, label)
    
    myTestFile = read_dataset(sys.argv[2])
    test(myTestFile, clf, sys.argv[3])

