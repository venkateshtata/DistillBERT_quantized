from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch


tokenizer = AutoTokenizer.from_pretrained("./distill_bert_imdb/checkpoint-782")
model = AutoModelForSequenceClassification.from_pretrained("./distill_bert_imdb/checkpoint-782")


dataset = load_dataset("imdb")
test_dataset = dataset['test']


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


test_dataset = tokenized_test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")


from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=8)


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
            print("predictions: ", pr)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            print("batch done.")
    return correct / total

print("evaluating now..")
accuracy = evaluate(test_dataloader)
print("Accuracy on test dataset:", accuracy)
