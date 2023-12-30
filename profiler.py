import torch
from torch.autograd import profiler
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.to("cuda:0")

input_text = "Hello, world! This is a test for BERT model."
encoded_input = tokenizer(input_text, return_tensors='pt')

encoded_input.to("cuda:0")

model.eval()

with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    with profiler.record_function("model_inference"):
        output = model(**encoded_input)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
else:
    print("CUDA is not available. Memory profiling is only for GPU.")
