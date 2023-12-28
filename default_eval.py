from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Subset


tokenizer = AutoTokenizer.from_pretrained("../models/distill_bert_imdb/checkpoint-782")
model = AutoModelForSequenceClassification.from_pretrained("../models/distill_bert_imdb/checkpoint-782")


dataset = load_dataset("imdb")
test_dataset = dataset['test']


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


test_dataset = tokenized_test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")


indices = torch.arange(1000).tolist()
subset_test_dataset = Subset(test_dataset, indices)

test_dataloader = DataLoader(subset_test_dataset, batch_size=256)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate(dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

print("evaluating now..")
accuracy = evaluate(test_dataloader)
print("Accuracy on test dataset:", accuracy)
