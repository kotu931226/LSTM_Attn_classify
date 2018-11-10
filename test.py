import torch
from models import ClassifyRNN
import config
from utils import DataOperat

num_ids = config.num_ids
emb_size = config.emb_size
hidden_size = config.hidden_size
n_classes = config.n_classes
device = config.device
y_path = config.y_path
pad_id_path = config.pad_id_path
batch_size = config.batch_size
save_model_path = config.save_model_path

model = ClassifyRNN(num_ids, emb_size, hidden_size, n_classes, device=device)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
data_set = DataOperat.create_data_set(pad_id_path, y_path, device)
data_set = data_set[:64*3]

correct = 0
total = 0
print('test')
for x, y in data_set:
    model.eval()
    output, _attention = model(x.unsqueeze(0))
    _, pred_ids = torch.max(output.data, 1)
    total += 1
    correct += 1 if pred_ids == y else 0
    print(x, int(y), int(pred_ids), output.to('cpu'))
print(f'Accuracy of all : {correct / total}')


class_correct = list(0. for i in range(n_classes))
class_total = list(0. for i in range(n_classes))
for x, y in data_set:
    output, _attention = model(x.unsqueeze(0))
    _, pred_ids = torch.max(output.data, 1)
    correct_ = 1 if int(pred_ids) == int(y) else 0
    for i in range(n_classes):
        y_idx = y
        class_correct[y_idx] += correct_
        class_total[y_idx] += 1

for i in range(n_classes):
    if class_correct[i] < 1 or class_total[i] < 1:
        continue
    print(f'Accuracy of {i} : {class_correct[i]/class_total[i]}')
