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
from torch.quantization import get_default_qconfig, prepare, convert
from torch.nn import LayerNorm, Embedding, Linear



tokenizer = AutoTokenizer.from_pretrained("../models/distill_bert_imdb/checkpoint-782")
model = AutoModelForSequenceClassification.from_pretrained("../models/distill_bert_imdb/checkpoint-782")

device = "cpu"
model.to(device)

model.eval()

# Define and apply a custom quantization configuration
default_qconfig = get_default_qconfig('fbgemm')

# Apply the quantization configuration to the model
for name, module in model.named_modules():
    if isinstance(module, LayerNorm) or isinstance(module, Linear):

        # Skip quantizing layer Normalization & Linear layers
        module.qconfig = None
    elif isinstance(module, Embedding):
        # Use a special configuration for embedding layers
        module.qconfig = float_qparams_weight_only_qconfig
    else:
        # Apply default configuration to other layers
        module.qconfig = default_qconfig
        

        

# Prepare the model for static quantization
model_fp32_prepared = prepare(model, inplace=False)
    


dataset = load_dataset("imdb")
test_dataset_raw = dataset['test']

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_test_dataset = test_dataset_raw.map(tokenize_function, batched=True)

test_dataset = tokenized_test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")

indices = torch.arange(100).tolist()
subset_test_dataset = Subset(test_dataset, indices)
test_dataloader = DataLoader(subset_test_dataset, batch_size=1)


print("collecting calibration values..")
sample_texts = []
count = 0
for s in tqdm(test_dataset_raw):
    if count==100:
        break
    sample_texts.append(s['text'])
    count+=1
    break

calib_inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    model_fp32_prepared(**calib_inputs)

model_int8 = convert(model_fp32_prepared, inplace=False)


# torch.save(model_int8.state_dict(), "../models/static_quantized_distillbert.pth")


def evaluate(dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model_int8(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

print("evaluating now..")
accuracy = evaluate(test_dataloader)
print("Accuracy on test dataset:", accuracy)
