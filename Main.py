from Config import parse_args
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Dictionary,Prepare_dict,DataSet,Trainer,Save_Loss,Save_All_Loss,seq_to_index,Load_Vocabs
from model import Att_Bi,Att_SumBiGRU,LeakyGRU,SLSTM,SGRU,BiSGRU,BiSLSTM
from manager_torch import GPUManager
def Train_Main(ModelName):
    args = parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    if not os.path.exists(args.save):
    	os.mkdir(args.save)
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # define the data directory
    files = ['SICK_train.txt','SICK_test_annotated.txt','SICK_trial.txt']
    train_dir = args.data + files[0]
    test_dir = args.data + files[1]
    dev_dir = args.data + files[2]
    # preparing the dictionary
    dictionary = Dictionary()
    vocab_file = "vocabcased.txt"
    if os.path.exists(vocab_file):
        Load_Vocabs(vocab_file,dictionary)
    else:
        for file in files:
            Prepare_dict(os.path.join(args.data,file),dictionary)
        dictionary.Save(vocab_file)
    # preparing the dataset
    train_dataset = DataSet(train_dir,dictionary)
    test_dataset = DataSet(test_dir,dictionary)
    trail_dataset = DataSet(dev_dir,dictionary)
    # some define of the model
    if ModelName == "Att-BiLSTM":
        model = Att_Bi(args,dictionary,"lstm")
    elif ModelName == "Att-BiGRU":
        model = Att_Bi(args,dictionary,"gru")
    elif ModelName == "LeakyGRU":
        model = LeakyGRU(args,dictionary)
    elif ModelName == "SGRU":
        model = SGRU(args,dictionary)
    elif ModelName == "SLSTM":
        model = SLSTM(args,dictionary)
    elif ModelName == "BiSLSTM":
        model = BiSLSTM(args,dictionary)
    elif ModelName == "BiSGRU":
        model = BiSGRU(args,dictionary)
    else:
        model = Att_SumBiGRU(args,dictionary)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    root_loss = model.__name__()
    TrianSet = Trainer(model,loss_function,optimizer,train_dataset,test_dataset,trail_dataset,args.epochs,root_loss)
    # training processing 
    all_losses,peason_list,mse_list = TrianSet.Train()
    # loss saved files
    loss_files = ["LossGlobal.txt","PeasonGlobal.txt","MSEGlobal.txt"]
    if not os.path.exists(root_loss):
        os.mkdir(root_loss)
    Save_All_Loss(all_losses,os.path.join(root_loss,loss_files[0]))
    Save_Loss(peason_list,os.path.join(root_loss,loss_files[1]))
    Save_Loss(mse_list,os.path.join(root_loss,loss_files[2]))
    # Test the Sentences
    Test_Single(TrianSet,os.path.join(root_loss,"Test_Sent.txt"))
def Test_Single(TrianSet,file_path):
    sent1 = ["A group of boys in a yard is playing and a man is standing in the background",
            "The young boys are playing outdoors and the man is smiling nearby",
            3.7]
    sent2 = ["A brown dog is attacking another animal in front of the tall man in pants",
            "A brown dog is attacking another animal in front of the man in pants",
            4.9]
    sent3 = ["A boy is playing a game with wooden blocks",
            "Some cheerleaders are dancing",
            1]
    Sentences = [sent1,sent2,sent3]
    value = []
    for index in range(len(Sentences)):
        sent = Sentences[index]
        sentA = sent[0].split()
        sentB = sent[1].split()
        sentA = seq_to_index(sentA,TrianSet.train_dataset.dictionary.word2idx)
        sentB = seq_to_index(sentB,TrianSet.train_dataset.dictionary.word2idx)
        hidden = TrianSet.model.initial_hidden()
        predict = TrianSet.model(sentA,sentB,hidden)
        if torch.cuda.is_available():
            predict = predict.cpu()
        value.append(predict.data.numpy()[0])
    with open(file_path,"w") as file:
        for index in range(len(Sentences)):
            temp = Sentences[index][0] + "\t" + Sentences[index][1] + "\t" + str(Sentences[index][2]) + "\t" + str(value[index])+ "\n"
            file.write(temp)
    print("Test Sentences done!")
def main():
    # Train_Main("Att-BiLSTM")
    # Train_Main("Att-BiGRU")
    # Train_Main("Att-SumBiGRU")
    # Train_Main("LeakyGRU")
    Train_Main("BiSLSTM")
    Train_Main("BiSGRU")
if __name__ == "__main__":
    main()
