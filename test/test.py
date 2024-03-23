import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "model/wishper", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,use_flash_attention_2=True
)
model.to("cuda:0")

processor = AutoProcessor.from_pretrained("model/wishper")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=512,
    generate_kwargs={"language": "english","task":"transcribe"},
    chunk_length_s=30,
    batch_size=32,
    return_timestamps=True,
    torch_dtype=torch.float16,
    device="cuda:0"
)


result = pipe("D0420-S1-T01.wav")
print(result["chunks"])
