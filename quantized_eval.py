import torch
import time
import collections
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.quantization
from torch.quantization import get_default_qconfig, QConfig
import transformers
from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig, float_qparams_weight_only_qconfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("../models/distill_bert_imdb/checkpoint-782")
model = AutoModelForSequenceClassification.from_pretrained("../models/distill_bert_imdb/checkpoint-782")
    

model.qconfig = default_dynamic_qconfig

# Special configuration for embedding layers
for name, module in model.named_modules():
    module.qconfig = float_qparams_weight_only_qconfig
    torch.quantization.quantize_dynamic(module, dtype=torch.qint8, inplace=True)
    
dataset = load_dataset("imdb")
test_dataset = dataset['test']


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


test_dataset = tokenized_test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")


indices = torch.arange(100).tolist()
subset_test_dataset = Subset(test_dataset, indices)

test_dataloader = DataLoader(subset_test_dataset, batch_size=1)


device = "cpu"
model.to(device)

def evaluate(dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
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



# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from datasets import load_dataset
# import torch
# 
# import transformers
# # from transformers import BertTokenizer, BertModel
# import torch.quantization
# from torch.quantization import get_default_qconfig, QConfig
# from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig, float_qparams_weight_only_qconfig
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
