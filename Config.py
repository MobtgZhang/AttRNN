import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Similarity calculation for sentence relatedess')
    #
    parser.add_argument('--data', default='data/sick/',
                        help='path to dataset')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    # model arguments
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Size of input word vector embeddings')
    # These arguments are for model Att-BiLSTM and Att-BiGRU
    parser.add_argument('--hidden_in_dim', default=200, type=int,
                        help='Size of first MLP classifier (Att-BiLSTM/Att-BiGRU)')
    parser.add_argument('--hidden_out_dim', default=100, type=int,
                        help='Size of second MLP classifier (Att-BiLSTM/Att-BiGRU)')
    parser.add_argument('--bi_temp_dim', default=50, type=int,
                        help='Size of out MLP classifier (Att-BiLSTM/Att-BiGRU)')
    # These argments are for model Att-SumBiGRU
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='Size of first MLP classifier (Att-SumBiGRU)')
    parser.add_argument('--sum_temp_dim', default=100, type=int,
                        help='Size of first MLP classifier (Att-SumBiGRU)')
    # training arguments
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    '''
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    '''
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    args = parser.parse_args()
    return args
