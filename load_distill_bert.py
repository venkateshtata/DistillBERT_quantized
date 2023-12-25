import psutil
import torch
# from transformers import AutoModel, AutoTokenizer
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import bitsandbytes as bnb

def quantize_model(model):
    """
    Quantize the weights of the model to 8-bit using bitsandbytes.
    Args:
    model (torch.nn.Module): The model to quantize.
    """
    for name, param in model.named_parameters():
        # Quantize only the weights, not the biases
        if "weight" in name:
            # Replace the standard PyTorch 32-bit parameter with an 8-bit version
            setattr(model, name, bnb.optim.GlobalQuantization.float32_to_uint8(param))


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

quantize_model(model)

print("model quantized!")

if torch.cuda.is_available():
    print("GPU is available")
    model = model.cuda()
    
input_text = "This is a sample text for profiling."
inputs = tokenizer(input_text, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}
    #print("inputs: ", inputs)

try:
    while True:
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    print("Exiting script...")





start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    #print("outputs: ", outputs)

end_time = time.time()

print(f"Time taken for forward pass: {end_time - start_time:.2f} seconds")

# Memory usage
print_memory_usage()
print_gpu_usage()

    
# pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("i hate muslims"))
