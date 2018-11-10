import torch

num_ids = 16
emb_size = 8
hidden_size = 32
n_classes = 4+1
n_epoch = 100
batch_size = 64

x_path = './data/Arithmetic_x.csv'
y_path = './data/Arithmetic_y.csv'
pad_id_path = './data/Arithmetic_pad_x.csv'
save_model_path = './classify.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
