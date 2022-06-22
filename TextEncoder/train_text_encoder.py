'''
These codes are revised by Hanwen from https://github.com/zhengyanzhao1997/NLP-model/blob/main/model/model/Torch_model/SimCSE-Chinese/train_unsupervised.py
April, 2022
'''
import json
import random
import collections
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import scipy.stats
from tqdm import tqdm
import os


# the path where you save the model
save_model = 'model/text_encoder.pth'
# the path where you store the data
graphine_repo = '/data/xuhw/data/Graphine/dataset/'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
tokenizer = AutoTokenizer.from_pretrained(bert_name)
Config = AutoConfig.from_pretrained(bert_name)
Config.attention_probs_dropout_prob = 0.3
Config.hidden_dropout_prob = 0.3

output_way = 'pooler'
assert output_way in ['pooler', 'cls', 'avg']


def read_graphine_texts_to_dict(path):
    texts = dict()
    name_lines = open(path+'/name.txt').readlines()
    def_lines = open(path + '/def.txt').readlines()
    for i in range(len(name_lines)):
        texts[name_lines[i].strip()] = name_lines[i].strip() + '. ' + def_lines[i].strip()
    return texts


def read_graphine_names(path):
    texts = collections.OrderedDict()
    name_lines = open(path+'/name.txt').readlines()
    for i in range(len(name_lines)):
        texts[name_lines[i].strip()] = 0
    return texts


def read_graph_to_dict(path):
    graph = json.load(open(path+'/graph.json', 'r', encoding='utf8'))
    return graph


file_list = os.listdir(graphine_repo)
file_exclude = ['go', 'cl']
graphin_emb = []
graphine_texts = []
test_texts = []
exclude_name, exlude_n = collections.OrderedDict(), 0
for file in file_exclude:
    graph = read_graph_to_dict(graphine_repo + file)
    text = read_graphine_names(graphine_repo + file)
    exclude_name.update(text)
for file in tqdm(file_list):
    graph = read_graph_to_dict(graphine_repo + file)
    text = read_graphine_texts_to_dict(graphine_repo + file)
    if file in file_exclude:
        for n1 in graph.keys():
            for n2 in graph[n1]:
                test_texts.append((text[n1.strip()], text[n2.strip()]))
        continue
    for n1 in graph.keys():
        for n2 in graph[n1]:
            if n1.strip() in exclude_name.keys() or n2.strip() in exclude_name.keys():
                exlude_n += 1
                continue
            graphine_texts.append((text[n1.strip()], text[n2.strip()]))
print('Exclude {} edges related to GO or CL!'.format(exlude_n))
random.shuffle(graphine_texts)
simCSE_data = graphine_texts.copy()
test_data = test_texts.copy()
print(len(simCSE_data))


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        sample = self.tokenizer([source[0], source[1]], max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name, config=Config)
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output


model = NeuralNetwork(bert_name, output_way).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

batch_size = 16
maxlen = 256
print('batch size: {} max length: {}'.format(batch_size, maxlen))
training_data = TrainDataset(simCSE_data, tokenizer, maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_data = TrainDataset(test_data, tokenizer, maxlen)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


def compute_corrcoef(x, y):
    """Spearman
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device='cuda')
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device='cuda') * 1e12
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def test(test_data, model):
    traget_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids, traget_attention_mask, traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred, source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = compute_corrcoef(label_list, similarity_list)
    return corrcoef


def train(dataloader, model, optimizer, save_path):
    model.train()
    size = len(dataloader.dataset)
    max_corrcoef = 0
    for batch, data in tqdm(enumerate(dataloader)):
        input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(device)
        pred = model(input_ids, attention_mask, token_type_ids)
        loss = compute_loss(pred)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * int(len(input_ids) / 2)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % 100 == 0:
            for test_batch, test_d in enumerate(test_dataloader):
                with torch.no_grad():
                    input_ids = test_d['input_ids'].view(len(test_d['input_ids']) * 2, -1).to(device)
                    attention_mask = test_d['attention_mask'].view(len(test_d['attention_mask']) * 2, -1).to(device)
                    token_type_ids = test_d['token_type_ids'].view(len(test_d['token_type_ids']) * 2, -1).to(device)
                    pred = model(input_ids, attention_mask, token_type_ids)
                    loss = compute_loss(pred)
                loss, current = loss.item(), batch * int(len(input_ids) / 2)
                print(f"Test loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                break
    torch.save(model.state_dict(), save_path)
    print("Saved PyTorch Model State to {}".format(save_path))


if __name__ == '__main__':
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer, save_model)
    print("Train_Done!")