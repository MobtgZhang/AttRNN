from torch.autograd import Variable
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
def Save_Loss(Matrix_All,file_path):
    with open(file_path,"w") as file:
        file.write("Train\tTest\tTrail\n")
        for index in range(len(Matrix_All)):
            file.write(str(Matrix_All[index][0])+"\t"+str(Matrix_All[index][1])+"\t"+str(Matrix_All[index][2])+"\n")
    print("Saved the file:%s"%file_path)
def Save_All_Loss(Matrix_All,file_path):
    with open(file_path,"w") as file:
        for index in range(len(Matrix_All)):
            file.write(str(Matrix_All[index])+"\n")
    print("Saved the file:%s"%file_path)
def Load_Vocabs(file_path,dictionary):
    print("Found %s,now loading the dictionary ......"%file_path)
    with open(file_path,"r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            dictionary.add_word(line)
    print("Successfully loaded words!")
class Trainer:
    def __init__(self,model,loss_func,optimizer,train_dataset,test_dataset,trail_dataset,epoches,save_file):
        self.model = model
        self.loss_function = loss_func
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.trail_dataset = trail_dataset
        self.epoches = epoches
        self.save_file = save_file
    def Train(self):
        Length = len(self.train_dataset)
        all_losses = []
        peason_list = []
        mse_list = []
        for epoch in tqdm(range(self.epoches),desc = "Epoch:"):
            TempSum = 0
            for index in tqdm(range(Length)):
                train_value = self.train_dataset[index]
                sentA = train_value[0]
                sentB = train_value[1]
                score = train_value[2]
                target_score = Variable(torch.FloatTensor([score]).view(1, 1))
                if torch.cuda.is_available():
                    sentA = sentA.cuda()
                    sentB = sentB.cuda()
                    target_score = target_score.cuda()
                self.optimizer.zero_grad()
                hidden = self.model.initial_hidden()
                predict_score = self.model(sentA,sentB,hidden)
                loss = self.loss_function(predict_score,target_score)
                loss.backward()
                self.optimizer.step()
                TempSum += loss.data[0]
            all_losses.append(TempSum/Length)
            # save model
            if not os.path.exists(self.save_file):
                os.mkdir(self.save_file)
            model_file = os.path.join(self.save_file,self.model.__name__() + ".pt")
            torch.save(self.model,model_file)
            # test Data
            x_data ,y_data = self.Test_test(self.train_dataset)
            Pea1 = Peason(x_data,y_data)
            Mse1 = MSE_Loss(x_data,y_data)
            x_data ,y_data = self.Test_test(self.test_dataset)
            Pea2 = Peason(x_data,y_data)
            Mse2 = MSE_Loss(x_data,y_data)
            x_data ,y_data = self.Test_test(self.trail_dataset)
            Pea3 = Peason(x_data,y_data)
            Mse3 = MSE_Loss(x_data,y_data)

            peason_list.append([Pea1,Pea2,Pea3])
            mse_list.append([Mse1,Mse2,Mse3])
        return all_losses,peason_list,mse_list
    def Test_test(self,dataset):
        Length = len(dataset)
        x_data = []
        y_data = []
        print("testing ...")
        for index in tqdm(range(Length)):
            train_value = dataset[index]
            sentA = train_value[0]
            sentB = train_value[1]
            score = train_value[2]
            target_score = Variable(torch.FloatTensor([score]).view(1, 1))
            if torch.cuda.is_available():
                sentA = sentA.cuda()
                sentB = sentB.cuda()
                target_score = target_score.cuda()
            hidden = self.model.initial_hidden()
            predict_score = self.model(sentA,sentB,hidden)
            predict_score = predict_score.cpu()
            x_data.append(target_score.data[0].numpy())
            y_data.append(predict_score.data[0].numpy())
        return np.array(x_data),np.array(y_data)
    def Test_Sentence(self,sentA,sentB):
        sentA = sentA.split()
        sentB = sentB.split()
        sentA = seq_to_index(sentA,self.train_dataset.dictionary)
        sentB = seq_to_index(sentB,self.train_dataset.dictionary)
        hidden = self.model.initial_hidden()
        score = self.model(sentA,sentB,hidden)
        return score.data.numpy()[0]
class Test:
    def __init__(self,model,dictionary):
        self.dictionary = dictionary
        self.model = model
    def Test_Sentence(self,sentA,sentB):
        hidden = self.model.initial_hidden()
        score = self.model(sentA,sentB,hidden)
        return score.data.numpy()[0]
# Make a dictionary
class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
    def Save(self,file_path):
        with open(file_path,"w") as file:
            for index in range(len(self.idx2word)):
                file.write(self.idx2word[index] + "\n")
        print("dictionary saved!")
    def __len__(self):
        return len(self.idx2word)
    def load_words(self,filename):
        print("Found words file: %s"%filename)
        with open(filename,"r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                self.add_word(line)
        print("Loaded the words !")
# load the words
def Prepare_dict(path_file,dictionary):
    data = pd.read_csv(path_file,sep="\t")
    Length = len(data)
    print("Preparing dictionary ......")
    FileName = os.path.split(path_file)
    for index in tqdm(range(Length),desc = FileName[1]):
        sentA = data.iloc[index]['sentence_A'].split()
        for word in sentA:
            dictionary.add_word(word)
        sentB = data.iloc[index]['sentence_B'].split()
        for word in sentB:
            dictionary.add_word(word)
# sentence to index sequence
def seq_to_index(sentence,word_to_index):
    ids = [word_to_index[word] for word in sentence]
    tensor = torch.LongTensor(ids)
    return Variable(tensor)
class DataSet:
    def __init__(self,data_path,dictionary):
        self.data_path = data_path
        self.sentences = []
        self.dictionary = dictionary
        print("Preparing DataSet ......")
        self.get_sentences()
        print("Dataset %s done!"%self.data_path)
    def get_sentences(self):
        dataset = pd.read_csv(self.data_path,sep = "\t")
        Length = len(dataset)
        FileName = os.path.split(self.data_path)
        for index in tqdm(range(Length),desc = FileName[1]):
            sentA = dataset.iloc[index]['sentence_A'].split()
            sentB = dataset.iloc[index]['sentence_B'].split()
            score = dataset.iloc[index]['relatedness_score']
            # indexing the word
            sFirst = seq_to_index(sentA,self.dictionary.word2idx)
            sSecond = seq_to_index(sentB,self.dictionary.word2idx)
            sScore = (float(score) - 1)/4
            temp = [sFirst,sSecond,sScore]
            self.sentences.append(temp)
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self,index):
        return self.sentences[index]
if __name__ == "__main__":
    root = 'data/sick/'
    files = ['SICK_test_annotated.txt','SICK_train.txt','SICK_trial.txt']
    dictionary = Dictionary()
    for file in files:
       Prepare_dict(os.path.join(root,file),dictionary)
    dictionary.Save("vocabcased.txt")
    for file in files:
        tempDataset = DataSet(os.path.join(root,file),dictionary)