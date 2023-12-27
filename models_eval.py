import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np



class SampleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return inputs, label


def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            print("batch shape: ", batch[0].items())
            inputs = {k: v.squeeze(0) for k, v in batch[0].items()}
            labels = batch[1]
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy



model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["This is a positive sentence.", "This is a negative sentence."]
labels = [1, 0]

dataset = SampleDataset(texts, labels, tokenizer)
data_loader = DataLoader(dataset, batch_size=2)

# state_dict = torch.load('normal.pth')
# model.load_state_dict(state_dict)

accuracy = evaluate(model, data_loader)
print(f"Accuracy: {accuracy}")