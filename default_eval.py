from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Subset
import transformers
import torch.quantization
from torch.quantization import get_default_qconfig, QConfig
from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig, float_qparams_weight_only_qconfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import time
import tracemalloc

device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("../models/distill_bert_imdb/checkpoint-782")
model = AutoModelForSequenceClassification.from_pretrained("../models/distill_bert_imdb/checkpoint-782")

torch.save(model.state_dict(), "../models/default_distill_bert.pth")

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


model.to(device)

def evaluate(dataloader):
    model.eval()
    total = 0
    correct = 0
    total_inference_time = 0
    total_cuda_memory_allocated = 0
    total_cuda_memory_reserved = 0
    num_batches = len(dataloader)

    start_time = time.time()  # Start timing

    # Start tracking memory allocation if not using CUDA
    if not torch.cuda.is_available():
        tracemalloc.start()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            inference_start_time = time.time()  # Time for each inference
            outputs = model(**inputs)
            total_inference_time += time.time() - inference_start_time

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Accumulate CUDA memory usage if available
            if torch.cuda.is_available():
                total_cuda_memory_allocated += torch.cuda.memory_allocated()
                total_cuda_memory_reserved += torch.cuda.memory_reserved()

    duration = time.time() - start_time  # Total evaluation time
    print(f"Total Evaluation Time: {duration:.4f} seconds")

    # Calculate and print average inference time and CUDA memory usage
    avg_inference_time = total_inference_time / num_batches
    print(f"Average Inference Time per Batch: {avg_inference_time:.4f} seconds")

    if torch.cuda.is_available():
        avg_memory_allocated = total_cuda_memory_allocated / num_batches / (1024 ** 2)  # Convert to MB
        avg_memory_reserved = total_cuda_memory_reserved / num_batches / (1024 ** 2)  # Convert to MB
        print(f"Average CUDA Memory Allocated per Batch: {avg_memory_allocated:.2f} MB")
        print(f"Average CUDA Memory Reserved per Batch: {avg_memory_reserved:.2f} MB")

    # Print memory usage if not using CUDA
    if not torch.cuda.is_available():
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current Memory Usage: {current / (1024 ** 2):.2f} MB")  # Convert to MB
        print(f"Peak Memory Usage: {peak / (1024 ** 2):.2f} MB")  # Convert to MB
        tracemalloc.stop()
    accuracy = correct / total
    print(f"Accuracy on test dataset: {accuracy:.6f}")


print("evaluating now..")
evaluate(test_dataloader)

