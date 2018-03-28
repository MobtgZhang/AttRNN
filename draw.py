import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from model import Att_Bi
from utils import Dictionary,Load_Vocabs,Test,DataSet
def get_allloss(filename):
    val_list = []
    with open(filename,"r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            val_list.append(float(line))
    return np.array(val_list)
def draw_other(file):
    paths = ["SLSTM","SGRU","BiSGRU","BiSLSTM"]
    train_list = []
    test_list = []
    trail_list = []
    for path in paths:
        filename = os.path.join(path,file)
        data = pd.read_csv(filename,sep="\t")
        data = data.as_matrix()
        train_list.append(data[:,0])
        test_list.append(data[:,1])
        trail_list.append(data[:,2])
    return train_list,test_list,trail_list
def draw_global():
    paths = ["SLSTM","SGRU","BiSGRU","BiSLSTM"]
    global_loss = []
    for path in paths:
        filename = os.path.join(path,"LossGlobal.txt")
        temp = get_allloss(filename)
        global_loss.append(temp)
    for index in range(len(global_loss)):
        x = np.linspace(0,len(global_loss[index])-1,len(global_loss[index]))
        y = global_loss[index]
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("Loss")
    plt.title("The global loss show")
    plt.legend(loc = "best")
    root = "result"
    if not os.path.exists(root):
        os.mkdir(root)
    plt.savefig(os.path.join(root,"Global.png"))
    plt.show()
def draw_MSE():
    paths = ["SLSTM","SGRU","BiSGRU","BiSLSTM"]
    root = "result"
    if not os.path.exists(root):
        os.mkdir(root)
    train_list,test_list,trail_list = draw_other("MSEGlobal.txt")
    for index in range(len(paths)):
        y = train_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("MSE Loss")
    plt.title("The MSE loss show for train")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Train_MSE.png"))
    plt.show()

    for index in range(len(paths)):
        y = test_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("MSE Loss")
    plt.title("The MSE loss show for test")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Test_MSE.png"))
    plt.show()

    for index in range(len(paths)):
        y = trail_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("MSE Loss")
    plt.title("The MSE loss show for trail")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Trail_MSE.png"))
    plt.show()
    data_list = []
    temp = []
    for index in range(len(train_list)):
        temp.append(train_list[index].min())
    data_list.append(temp);temp = []
    for index in range(len(test_list)):
        temp.append(test_list[index].min())
    data_list.append(temp);temp = []
    for index in range(len(trail_list)):
        temp.append(trail_list[index].min())
    data_list.append(temp);temp = []
    return data_list
def draw_Peason():
    paths = ["SLSTM","SGRU","BiSGRU","BiSLSTM"]
    root = "result"
    if not os.path.exists(root):
        os.mkdir(root)
    train_list,test_list,trail_list = draw_other("PeasonGlobal.txt")
    for index in range(len(paths)):
        y = train_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("Peason Loss")
    plt.title("The Peason loss show for train")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Train_Peason.png"))
    plt.show()

    for index in range(len(paths)):
        y = test_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("Peason Loss")
    plt.title("The Peason loss show for test")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Test_Peason.png"))
    plt.show()

    for index in range(len(paths)):
        y = trail_list[index]
        x = np.linspace(0,len(y)-1,len(y))
        plt.plot(x,y,label = paths[index])
    plt.xlabel("Epoches ")
    plt.ylabel("Peason Loss")
    plt.title("The Peason loss show for trail")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(root,"Trail_Peason.png"))
    plt.show()
    data_list = []
    temp = []
    for index in range(len(train_list)):
        temp.append(train_list[index].max())
    data_list.append(temp);temp = []
    for index in range(len(test_list)):
        temp.append(test_list[index].max())
    data_list.append(temp);temp = []
    for index in range(len(trail_list)):
        temp.append(trail_list[index].max())
    data_list.append(temp);temp = []
    return data_list
def save_data(data_list,filename):
    list_label = ['train','test','trail']
    list_model = ["SLSTM","SGRU","BiSGRU","BiSLSTM"]
    with open(os.path.join("result",filename),"w") as file:
        for k in range(len(list_model)):
            file.write(list_model[k] + "\t")
        file.write("\n")
        for j in range(len(list_label)):
            file.write(list_label[j] + "\t")
            for k in range(len(data_list[j])):
                file.write(str(data_list[j][k]) + "\t")
            file.write("\n")
def show_trail():
    filename = "Att-BiGRU/Att-BiGRU.pt"
    model = torch.load(filename)
    dictionary = Dictionary()
    Load_Vocabs("vocabcased.txt",dictionary)
    Test_Sent = Test(model,dictionary)
    dataset = DataSet("data/sick/SICK_trial.txt",dictionary)
    with open("Refer_Set.txt","w") as file:
        for index in range(len(dataset)):
            sentences = dataset[index]
            predict = Test_Sent.Test_Sentence(sentences[0],sentences[1])
            predict = predict.data[0]
            real = sentences[2]
            file.write(str(real) + "\t" + str(predict) + "\n")
def main():
    draw_global()
    mse_list = draw_MSE()
    peason_list = draw_Peason()
    save_data(mse_list,"best_MSE.txt")
    save_data(peason_list,"best_Peason.txt")
    # show the last trail data
    show_trail()

if __name__ == '__main__':
    main()
