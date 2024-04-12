import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline


class Model:
    
    def __init__(self) -> None:
        pass
    
    def load_model():
        model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3",
                                                    torch_dtype=torch.float16,
                                                    attn_implementation="flash_attention_2",
                                                    use_safetensors=True,
                                                    low_cpu_mem_usage=False
                                                    ).to("cuda:0")
        
        return model

    def load_tokenizer():
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        return processor
    
    def pyannote_pipeline():
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="").to(torch.device("cuda:0"))
        return pipeline
