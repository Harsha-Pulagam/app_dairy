import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

"""
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

save_dir = "model/wishper"

model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
"""
