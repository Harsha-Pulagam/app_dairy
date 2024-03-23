from model import Model
import torch,torchaudio
from transformers import pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

class Inference:
    def __init__(self):
        self.model = Model.load_model()
        self.processor = Model.load_tokenizer()
        self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    max_new_tokens=256,
                    generate_kwargs={"language": "english","task":"transcribe"},
                    chunk_length_s=90,
                    batch_size=64,
                    return_timestamps=True,
                    torch_dtype=torch.float16,
                    device="cuda:0",
                )
        self.dairization = Model.pyannote_pipeline()
        
    def trascribe(self, audio_path):
        transcription = self.pipe(audio_path)
        return transcription["chunks"]
    
    def speaker_dairization(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        with ProgressHook() as hook:
            dairization = self.dairization({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
        return dairization


"""    
if __name__ == "__main__":
    audio_path = "D0420-S1-T01.wav"
    inference = Inference()
    transcription = inference.trascribe(audio_path)
    print(transcription)
    
"""