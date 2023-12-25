import psutil
import torch
# from transformers import AutoModel, AutoTokenizer
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
import torch.quantization


def print_memory_usage():
    # Get memory usage in GB
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 3)
    print(f"Memory usage: {memory_usage:.2f} GB")

def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory usage: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    else:
        print("No GPU available.")


model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

quantization_config = torch.quantization.get_default_qconfig('fbgemm')
model.qconfig = quantization_config

torch.quantization.prepare(model, inplace=True)

# Calibrate the model
def calibrate(model, data_loader):
    with torch.no_grad():
        for batch in data_loader:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            model(**inputs)

calibration_sentences = ["I love you", "I hate you"]
calibration_data_loader = DataLoader(calibration_sentences, batch_size=2)

calibrate(model, calibration_data_loader)


torch.quantization.convert(model, inplace=True)
print("model quantized!")

if torch.cuda.is_available():
    print("GPU is available")
    model = model.cuda()
    
input_text = "This is a sample text for profiling."
inputs = tokenizer(input_text, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}
    #print("inputs: ", inputs)

# try:
#     while True:
#         time.sleep(1)  # Sleep to reduce CPU usage
# except KeyboardInterrupt:
#     print("Exiting script...")



start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    #print("outputs: ", outputs)

end_time = time.time()

print(f"Time taken for forward pass: {end_time - start_time:.2f} seconds")

# Memory usage
print_memory_usage()
print_gpu_usage()
