from gensim import models
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def Peason(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    x = x - x.mean()
    y = y - y.mean()
    return x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
def MSE_Loss(x,y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    Tmp = x - y
    Tmp = np.power(Tmp,2)
    return Tmp.mean()
class IterSentences:
	def __init__(self,dirname):
		self.dirname = dirname
	def __iter__(self):
		for line in open(self.dirname):
			yield line.split()
def save_sentences(filename):
	print("saving sentences ...")
	dirnames = ["G:\\Paper_All\\Code\\data\\sick\\SICK_train.txt",
				"G:\\Paper_All\\Code\\data\\sick\\SICK_test_annotated.txt",
				"G:\\Paper_All\\Code\\data\\sick\\SICK_trial.txt"]
	with open(filename,"w") as f:
		for file in dirnames:
			OutData = pd.read_csv(file,sep = "\t")
			Length = len(OutData)
			for index in range(Length):
				f.write(OutData.iloc[index]['sentence_A'] + "\n")
				f.write(OutData.iloc[index]['sentence_B'] + "\n")
	print("saved sentences!")
def train_model(dirname):
	sentences = IterSentences(dirname)
	print("training ...")
	Model = models.Word2Vec(sentences)
	print("trained model!")
	Model.save("word2vec.pt")
def get_sentence_embedding(Model,sentence):
	try:
		value = Model[sentence[0]]
	except KeyError:
		value=np.random.random(size=100)
	dimension = len(value)
	value = value.reshape(1,dimension)
	for index in range(1,len(sentence)):
		try:
			temp = Model[sentence[index]]
			value = np.vstack((value,temp))
		except KeyError:
			out=np.random.random(size=(1,dimension))
			value = np.vstack((value,out))
	return value
def cosine_value(valueA,valueB):
	out = valueA.dot(valueB)
	lenA = np.linalg.norm(valueA,ord=2)
	lenB = np.linalg.norm(valueB,ord=2)
	return out/(lenA*lenB)
def similarity_sent(line1,line2):
	Num = 0
	Sum = 0
	for k in range(len(line1)):
		for j in range(len(line2)):
			out = cosine_value(line1[k],line2[j])
			if out > 0:
				Num += 1
				Sum += out
	value = Sum/Num
	return value
def calculate_sentences(filename,modelfile):
	OutData = pd.read_csv(filename,sep = "\t")
	Length = len(OutData)
	Model = models.Word2Vec.load(modelfile)
	x = []
	y = []
	for index in tqdm(range(Length)):
		line1 = OutData.iloc[index]['sentence_A'].split()
		line2 = OutData.iloc[index]['sentence_B'].split()
		label = float(OutData.iloc[index]['relatedness_score'])
		label = (label -1)/4
		sentA = get_sentence_embedding(Model,line1)
		sentB = get_sentence_embedding(Model,line2)
		score = similarity_sent(sentA,sentB)
		x.append(label)
		y.append(score)
	x = np.array(x);y = np.array(y)
	valA = Peason(x,y)
	valB = MSE_Loss(x,y)
	return valA,valB
def main():
	save_file = "sentences.txt"
	if not os.path.exists(save_file):
		save_sentences(save_file)
	if not os.path.exists("word2vec.pt"):
		train_model(save_file)
	
	dirnames = ["G:\\Paper_All\\Code\\data\\sick\\SICK_train.txt",
				"G:\\Paper_All\\Code\\data\\sick\\SICK_test_annotated.txt",
				"G:\\Paper_All\\Code\\data\\sick\\SICK_trial.txt"]
	for file in dirnames:
		valA,valB = calculate_sentences(file,"word2vec.pt")
		print(valA,valB)
if __name__ == '__main__':
	main()