import torch
import time
import collections
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.quantization
from torch.quantization import get_default_qconfig, QConfig
import transformers
from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig, float_qparams_weight_only_qconfig



model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# torch.save(model.state_dict(), "normal.pth")

def collect_unique_module_types(model):
    module_types = collections.defaultdict(int)
    for module in model.modules():
        module_types[type(module)] += 1
    return module_types

# Collect unique module types
unique_module_types = collect_unique_module_types(model)

# Print the unique module types
print("Unique module types in the model:")
for module_type, count in unique_module_types.items():
    print(f"{module_type.__name__}: {count} instances")
    

model.qconfig = default_dynamic_qconfig

# Special configuration for embedding layers
for name, module in model.named_modules():
    module.qconfig = float_qparams_weight_only_qconfig
    torch.quantization.quantize_dynamic(module, dtype=torch.qint8, inplace=True)


# torch.save(model.state_dict(), "quant.pth")

text = ["Example text for quantized BERT."]
encoded_input = tokenizer(text, return_tensors='pt')


print(model)
