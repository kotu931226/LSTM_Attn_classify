# import random
from tqdm import tqdm
import torch
from torch import nn
from utils import DataOperat
from models import ClassifyRNN
import config

num_ids = config.num_ids
emb_size = config.emb_size
hidden_size = config.hidden_size
n_classes = config.n_classes
n_epoch = config.n_epoch
batch_size = config.batch_size
pad_id_path = config.pad_id_path
y_path = config.y_path
save_model_path = config.save_model_path

device = config.device
print('using: ', device)

model = ClassifyRNN(num_ids, emb_size, hidden_size, n_classes, device=device).to(device)
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
data_set = DataOperat.create_data_set(pad_id_path, y_path, device=device)
dev_idx = len(data_set)*7//8
loss_fn = nn.NLLLoss(ignore_index=0)

def train(model, data, optimizer, n_epoch, batch_size, dev_data=None):
    for epoch in range(1, n_epoch+1):
        model.train()
        print('-----Epoch: ', epoch)
        gen_batch_data = DataOperat.gen_batch_data(data, batch_size)
        for i, batch_data in enumerate(tqdm(gen_batch_data)):
            preds, _attentions = model(batch_data[0])
            loss = loss_fn(preds, batch_data[1])
            if i % 1000 == 0 and i != 0:
                print('  Epoch', epoch, 'loss', loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        test(model, dev_data, batch_size)

def test(model, dev_data, batch_size):
    model.eval()
    correct = 0
    count = 0
    losses = []
    gen_batch_data = DataOperat.gen_batch_data(dev_data, batch_size)
    for batch_data in gen_batch_data:
        preds, _attentions = model(batch_data[0])
        loss = loss_fn(preds, batch_data[1])
        losses.append(loss.item())
        _, pred_ids = torch.max(preds, 1)
        correct += torch.sum(pred_ids == batch_data[1]).item()
        count += batch_size
    print('-----Test Result-----')
    print('loss:', sum(losses) / count)
    print('accuracy:', correct / count)
    print()
            

train_data = data_set[:dev_idx]
dev_data = data_set[dev_idx:]

train(model, train_data, optimizer, n_epoch, batch_size, dev_data=dev_data)

torch.save(model.state_dict(), save_model_path)
