import torch
import time

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.quantization
from torch.quantization import get_default_qconfig, QConfig
import transformers


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.eval()

# Function to apply dynamic quantization to specific layers - encoder.layer.3.output.LayerNorm.weight
def quantize_attention_heads(model):
    for name, module in model.named_modules():
        # print("name: ", name)
        # print("module: ", module)
        # print("instance class: ", type(module))
        # print("==========")
        # Check if the module is part of the attention mechanism
        if name=="encoder.layer.11.intermediate.dense" and isinstance(module, torch.nn.modules.linear.Linear):
            # Quantize the module dynamically
            torch.quantization.quantize_dynamic(module, dtype=torch.qint8, inplace=True)

quantize_attention_heads(model)



text = ["Example text for quantized BERT."]
encoded_input = tokenizer(text, return_tensors='pt')

# with torch.no_grad():
#     quantized_output = quantized_model(**encoded_input)


# Call the function with the loaded model
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}\n")