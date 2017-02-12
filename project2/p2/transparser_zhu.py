#!/usr/bin/python
import sys

LEFT = 0
RIGHT = 1
SHIFT = 2

class Sentence():
    def __init__(self):
        root = {"index": "0", \
                "word": "*root*", \
                "coarsePOS": "*root*", \
                "parrentIndex": None}
        self.stack = [root]
        self.buffer = []
        self.arcs = []
	
    def addWord(self, line):
        line = line.split()
        index = line[0]
        word = line[1]
        coarsePOS = line[3]
        parrentIndex = line[6]
        self.buffer.append({"index": index, \
                            "word": word, \
                            "coarsePOS": coarsePOS, \
                            "parrentIndex": parrentIndex})

    def popBuffer(self):
        topWord = self.buffer[0]
        del self.buffer[0]
        return topWord

    # update the stack and buffer for a given transition
    def update(self, transition, bufferTopWord):
        if transition == LEFT:
            # when stack only contains root, left arc is not allowed

            self.arcs.append((bufferTopWord.get("index"), \
                            self.stack[-1].get("index")))
            del self.stack[-1]
            self.buffer.insert(0, bufferTopWord)
        elif transition == RIGHT:
            self.arcs.append((self.stack[-1].get("index"), \
                            bufferTopWord.get("index")))
            self.buffer.insert(0, self.stack[-1])
            del self.stack[-1]
        else:
            self.stack.append(bufferTopWord)

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
class perceptron():
    def __init__(self):
    	self.Weights = [Weights() for i in range(3)]
        self.counter = 0
        self.accuracy = [0, 0]

    def train(self, sentence):
        while sentence.buffer:
            if not sentence.stack:
                bufferTopWord = sentence.popBuffer()
                sentence.update(SHIFT, bufferTopWord)

            if not sentence.buffer:
                break
            bufferTopWord = sentence.popBuffer()

            # get ground truth
            assert bufferTopWord.get("parrentIndex") != '_'
            r_parrentIndex = bufferTopWord.get("parrentIndex")
            r_index = bufferTopWord.get("index")
            l_parrentIndex = sentence.stack[-1].get("parrentIndex")
            l_index = sentence.stack[-1].get("index")
            parentSet = set(map(lambda x: x.get("parrentIndex"), sentence.buffer))
            if r_parrentIndex == l_index and not r_index in parentSet:
                t_true = RIGHT
            elif l_parrentIndex == r_index and not l_index in parentSet:
                t_true = LEFT
            else:
                t_true = SHIFT

            #  update perceptron
            self.update(sentence.stack[-1], bufferTopWord, t_true)

            # update stack and buffer
            sentence.update(transition = t_true, bufferTopWord = bufferTopWord)

    def predict(self, sentence):
        while sentence.buffer:
            if not sentence.stack:
                bufferTopWord = sentence.popBuffer()
                sentence.update(SHIFT, bufferTopWord)

            if not sentence.buffer:
                break
            bufferTopWord = sentence.popBuffer()

            t = self.predictTransition(sentence.stack[-1], bufferTopWord)

            # update stack and buffer
            sentence.update(transition = t, bufferTopWord = bufferTopWord)

        return sentence.arcs

    def update(self, lWord, rWord, truth):
        pred = self.predictTransition(lWord, rWord)
        self.counter += 1
        self.accuracy[1] += 1
        if pred != truth:
            feats = self.getFeatures(lWord, rWord)
            self.Weights[truth].update(feats, 1./self.counter)
            self.Weights[pred].update(feats, -1./self.counter)
        else:
            self.accuracy[0] += 1


    def predictTransition(self, lWord, rWord):
        feats = self.getFeatures(lWord, rWord)
        lw = self.Weights[LEFT].dotProduct(feats)
        rw = self.Weights[RIGHT].dotProduct(feats)
        sw = self.Weights[SHIFT].dotProduct(feats)
        # print lWord.get("word"), rWord.get("word"), lw, rw, sw
        if lw > rw and lw > sw:
            return LEFT
        elif rw > sw:
            return RIGHT
        else:
            return SHIFT

    def getFeatures(self, lWord, rWord):
        feats = {}
        feats["stack:" + lWord.get("word")] = 1.
        feats["buffer:" + rWord.get("word")] = 1.
        feats["stackPOS:" + lWord.get("coarsePOS")] = 1.
        feats["bufferPOS:" + rWord.get("coarsePOS")] = 1.
        feats["word_pair:" + lWord.get("word") + "_" + rWord.get("word")] = 1.
        feats["POS_pair:" + lWord.get("coarsePOS") + "_" + rWord.get("coarsePOS")] = 1.
        return feats

    def resetAccuracy(self):
        self.accuracy = [0, 0]

    def returnAccuracy(self):
        return float(self.accuracy[0])/self.accuracy[1]

def readFile(fileName):
    with open(fileName, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                sentence.append(line)
            else:
                yield sentence
                sentence = []
        if sentence:
            yield sentence

def test():
    trainingFile = sys.argv[1]
    system = perceptron()
    for i in range(30):
        system.resetAccuracy()
        for sentence in readFile(trainingFile):
            s = Sentence()
            for word in sentence:
                s.addWord(word)
            system.train(s)
        print system.returnAccuracy()

    testingFile = sys.argv[2]
    result = ""
    acc = [0,0]
    for sentence in readFile(testingFile):
        s = Sentence()
        for word in sentence:
            s.addWord(word)
        arcs = system.predict(s)
        arcDict = {}
        for arc in arcs:
            arcDict[arc[1]] = arc[0]
        for word in sentence:
            tmp = word.split()
            acc[1] += 1
            if tmp[6] == arcDict.get(tmp[0], '0'):
                acc[0] += 1
            tmp[6] = arcDict.get(tmp[0], '0')
            result += '\t'.join(tmp) + '\n'
        result += '\n'
    outputFile = sys.argv[3]
    print acc
    with open(outputFile, "w") as f:
        f.write(result)


test()











