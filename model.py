import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import copy

class ContactLinear(nn.Module):
    def __init__(self,in_dimA,in_dimB,out_feature,bais = True):
        super(ContactLinear,self).__init__()
        self.in_dimA = in_dimA
        self.in_dimB = in_dimB
        self.out_feature = out_feature
        self.weightA = Parameter(torch.randn(in_dimA,out_feature))
        self.weightB = Parameter(torch.randn(in_dimB,out_feature))
        if bais:
            self.bias = Parameter(torch.randn(out_feature))
        else:
            self.reset_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        stdvA = 1./math.sqrt(self.weightA.size(1))
        stdvB = 1./math.sqrt(self.weightB.size(1))
        self.weightA.data.uniform_(-stdvA,stdvA)
        self.weightB.data.uniform_(-stdvB,stdvB)
        if self.bias is not None:
            self.bias.data.uniform_(-stdvB,stdvB)
    def forward(self,inputA,inputB):
        return F.linear(inputA,self.weightA.t(),None)+F.linear(inputB,self.weightB.t(),self.bias)
    def __repr__(self):
        return self.__class__.__name__ + ' ( ('\
            + str(self.in_dimA) + ' , '\
            + str(self.in_dimB) + ' ) ->'\
            + str(self.out_feature) + ' ) '
class Att_Bi(nn.Module):
    def __init__(self,params,dictionary,algorithm):
        super(Att_Bi,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)
        # embedding layer
        self.embedding_dim = params.embedding_dim
        self.hidden_in_dim = params.hidden_in_dim
        self.hidden_out_dim = params.hidden_out_dim
        self.out_dim = 1
        self.temp_dim = params.bi_temp_dim
        self.vocab_size = len(self.dictionary)
        self.algorithm = algorithm.strip().lower()
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if self.algorithm == "lstm":
            self.rnn = nn.LSTM(self.embedding_dim,self.hidden_in_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = True)
        elif self.algorithm == "gru":
            self.rnn = nn.GRU(self.embedding_dim,self.hidden_in_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = True)
        self.bi_linear = ContactLinear(self.hidden_in_dim,self.hidden_in_dim,self.hidden_out_dim)
        self.tanh = nn.Tanh()
        self.extract = ContactLinear(self.hidden_out_dim,self.hidden_out_dim,self.temp_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.temp_dim,self.out_dim)
        self.sigmoid = nn.Sigmoid()
        self.initial_hidden()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.rnn(embA,hidden)
        _,htB = self.rnn(embB,hidden)
        if self.algorithm == "lstm":
            htA = htA[0]
            htB = htB[0]
        htA = htA.squeeze()
        htB = htB.squeeze()
        ht = htA * htB
        hv = torch.abs(htA-htB)
        # make the differece and the multiply together        
        hs = self.bi_linear(hv,ht)
        hs = self.tanh(hs)
        # make two different directions together
        htA = hs[0].view(1,self.hidden_out_dim)
        htB = hs[1].view(1,self.hidden_out_dim)
        ht = self.extract(htA,htB)
        ht = self.relu(ht)
        out = self.out(ht)
        return self.sigmoid(out)
    def initial_hidden(self):
        if self.algorithm == "lstm":
            out =  (Variable(torch.zeros(2,1,self.hidden_in_dim)),Variable(torch.zeros(2,1,self.hidden_in_dim)))
        elif self.algorithm == "gru":
            out =  Variable(torch.zeros(2,1,self.hidden_in_dim))
        if torch.cuda.is_available():
            if self.algorithm == "lstm":
                out[0] = out[0].cuda()
                out[1] = out[1].cuda()
            elif self.algorithm == "gru":
                out = out.cuda()
        return out
    def __name__(self):
        return "Att-Bi" + self.algorithm.upper()
class Att_SumBiGRU(nn.Module):
    def __init__(self,params,dictionary):
        super(Att_SumBiGRU,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)        
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.temp_dim = params.sum_temp_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = True)
        self.ones_matrix = Variable(torch.ones(1,self.temp_dim))
        if torch.cuda.is_available():
            self.ones_matrix.cuda()
        self.linear_second = nn.Linear(self.hidden_dim,self.temp_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
        self.init_weight()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.gru(embA,hidden)
        _,htB = self.gru(embB,hidden)
        # Disperse
        ht_leftA = htA[0].view(1,self.hidden_dim)
        ht_leftB = htA[1].view(1,self.hidden_dim)
        ht_rightA = htB[0].view(1,self.hidden_dim)
        ht_rightB = htB[1].view(1,self.hidden_dim)
        #-------------------------------------------------------------------------------
        # The first hidden Layer Attention Layer
        hf = self.ones_matrix
        #-------------------------------------------------------------------------------
        # The second hidden Layer
        ht_MA = torch.abs(ht_leftA - ht_rightA)
        ht_MB = ht_leftA * ht_rightA
        htqA = torch.abs(ht_leftB - ht_rightB)
        htqB = ht_leftB * ht_rightB
        Hidden_Temp = torch.cat((ht_MA,ht_MB,htqA,htqB),dim = 0)
        HiddenB = self.linear_second(Hidden_Temp)
        hq = self.relu(HiddenB)
        # multiply the matrix
        hs = torch.mm(hf,hq.t())
        ht = self.linear(hs)
        return self.sigmoid(ht)
    def initial_hidden(self):
        out =  Variable(torch.zeros(2,1,self.hidden_dim))
        if torch.cuda.is_available():
            return out.cuda()
    def __name__(self):
        return "Att-SumBiGRU"

class LeakyGRU(nn.Module):
    def __init__(self,params,dictionary):
        super(LeakyGRU,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)     
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.temp_dim = params.sum_temp_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = False)
        self.linear_second = ContactLinear(self.hidden_dim,self.hidden_dim,self.temp_dim)
        self.linear = nn.Linear(self.temp_dim,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.gru(embA,hidden)
        _,htB = self.gru(embB,hidden)
        #-------------------------------------------------------------------------------
        htA = htA.view(1,self.hidden_dim)
        htB = htB.view(1,self.hidden_dim)
        hs = self.linear_second(htA,htB)
        ht = F.relu(hs)
        ht = self.linear(ht)
        return self.sigmoid(ht)
    def initial_hidden(self):
        out =  Variable(torch.zeros(2,1,self.hidden_dim))
        if torch.cuda.is_available():
            return out.cuda()
    def __name__(self):
        return "LeakyGRU"
class SLSTM(nn.Module):
    def __init__(self,params,dictionary):
        super(SLSTM,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)     
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = False)
        self.linear_second = ContactLinear(self.hidden_dim,self.hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.lstm(embA,hidden)
        _,htB = self.lstm(embB,hidden)
        #-------------------------------------------------------------------------------
        htA = htA[0].view(1,self.hidden_dim)
        htB = htB[1].view(1,self.hidden_dim)
        hs = self.linear_second(htA,htB)
        return self.sigmoid(hs)
    def initial_hidden(self):
        out =  (Variable(torch.zeros(1,1,self.hidden_dim)),Variable(torch.zeros(1,1,self.hidden_dim)))
        if torch.cuda.is_available():
            out[0].cuda()
            out[1].cuda()
            return out
    def __name__(self):
        return "SLSTM"
class SGRU(nn.Module):
    def __init__(self,params,dictionary):
        super(SGRU,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)     
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = False)
        self.linear_second = ContactLinear(self.hidden_dim,self.hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.gru(embA,hidden)
        _,htB = self.gru(embB,hidden)
        #-------------------------------------------------------------------------------
        htA = htA.view(1,self.hidden_dim)
        htB = htB.view(1,self.hidden_dim)
        hs = self.linear_second(htA,htB)
        return self.sigmoid(hs)
    def initial_hidden(self):
        out =  Variable(torch.zeros(1,1,self.hidden_dim))
        if torch.cuda.is_available():
            out.cuda()
            return out
    def __name__(self):
        return "SGRU"

class BiSGRU(nn.Module):
    def __init__(self,params,dictionary):
        super(BiSGRU,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)     
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = True)
        self.linear_second = ContactLinear(self.hidden_dim*2,self.hidden_dim*2,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.gru(embA,hidden)
        _,htB = self.gru(embB,hidden)
        #-------------------------------------------------------------------------------
        htA = htA.view(1,self.hidden_dim*2)
        htB = htB.view(1,self.hidden_dim*2)
        hs = self.linear_second(htA,htB)
        return self.sigmoid(hs)
    def initial_hidden(self):
        out =  Variable(torch.zeros(1,1,self.hidden_dim))
        if torch.cuda.is_available():
            out.cuda()
            return out
    def __name__(self):
        return "BiSGRU"

class BiSLSTM(nn.Module):
    def __init__(self,params,dictionary):
        super(BiSLSTM,self).__init__()
        self.dictionary = copy.deepcopy(dictionary)     
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.vocab_size = len(self.dictionary)
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # GRU Layer
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim,num_layers = 1,
                                bias = True,batch_first = True,dropout = 0.2,bidirectional = True)
        self.linear_second = ContactLinear(self.hidden_dim*2,self.hidden_dim*2,1)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.cuda()
    def forward(self,sentA,sentB,hidden):
        embA = self.embedding(sentA)
        embB = self.embedding(sentB)
        embA = embA.view(1,len(sentA),self.embedding_dim)
        embB = embB.view(1,len(sentB),self.embedding_dim)
        
        _,htA = self.lstm(embA,hidden)
        _,htB = self.lstm(embB,hidden)
        
        #-------------------------------------------------------------------------------
        htA = htA[0].view(1,self.hidden_dim*2)
        htB = htB[1].view(1,self.hidden_dim*2)
        hs = self.linear_second(htA,htB)
        return self.sigmoid(hs)
    def initial_hidden(self):
        out =  Variable(torch.zeros(1,1,self.hidden_dim))
        if torch.cuda.is_available():
            out.cuda()
            return out
    def __name__(self):
        return "BiSLSTM"
