import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline


class Model:
    
    def __init__(self) -> None:
        pass
    
    def load_model():
        model = AutoModelForSpeechSeq2Seq.from_pretrained("model/wishper",
                                                    torch_dtype=torch.float16,
                                                    attn_implementation="flash_attention_2",
                                                    use_safetensors=True,
                                                    low_cpu_mem_usage=False
                                                    ).to("cuda:0")
        
        return model

    def load_tokenizer():
        processor = AutoProcessor.from_pretrained("model/wishper")
        return processor
    
    def pyannote_pipeline():
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            min_speakers=2,
                                            max_speakers=5,
                                        use_auth_token="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").to(torch.device("cuda:0"))
        return pipeline