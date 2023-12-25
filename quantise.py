import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
import torch.quantization


model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
    break

# Inspect buffer data types
for name, buf in model.named_buffers():
    print(f"{name}: {buf.dtype}")
    break
    
print("===============================")

    
# Apply a general quantization configuration for linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare specific modules for quantization
quantize_modules = ['classifier', 'bert.pooler']
for name, module in model.named_children():
    if name in quantize_modules:
        torch.quantization.prepare(module, inplace=True)


input_text = ["i love you", "i hate you"]
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
# Convert the prepared modules to quantized version
for name, module in model.named_children():
    if name in quantize_modules:
        torch.quantization.convert(module, inplace=True)

    
# Inspect parameter data types
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")


# Inspect buffer data types
for name, buf in model.named_buffers():
    print(f"{name}: {buf.dtype}")

