from model import Model
import torch,torchaudio
from transformers import pipeline
from pyannote.audio import Pipeline
from pyannote.audio import Audio
import numpy as np




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
        return transcription
    
    def speaker_dairization(self, audio_path):
        io = Audio(mono="downmix", sample_rate=16000)
        waveform ,sample_rate = io(audio_path)
        dairization = self.dairization({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=2, max_speakers=5)
        return dairization
    
    def labeling_speakers(self, dairization,transcription):
        segments = []
        for segment, track, label in dairization.itertracks(yield_label=True):
            segments.append({'segment': {'start': segment.start, 'end': segment.end},
                                'track': track,
                                'label': label})
        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]

            # check if we have changed speaker ("label")
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                # add the start/end times for the super-segment to the new list
                new_segments.append(
                    {
                        "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                        "speaker": prev_segment["label"],
                    }
                )
                prev_segment = segments[i]

        # add the last segment(s) if there was no speaker change
        new_segments.append(
            {
                "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
                "speaker": prev_segment["label"],
            }
        )
        
        transcript = transcription["chunks"]

        # get the end timestamps for each chunk from the ASR output
        end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
        segmented_preds = []

        # align the diarizer timestamps and the ASR timestamps
        for segment in new_segments:
            # get the diarizer end timestamp
            end_time = segment["segment"]["end"]
            # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))

            if True:
                segmented_preds.append(
                    {
                        "speaker": segment["speaker"],
                        "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                        "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                    }
                )
            else:
                for i in range(upto_idx + 1):
                    segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

            # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
            transcript = transcript[upto_idx + 1 :]
            end_timestamps = end_timestamps[upto_idx + 1 :]
            
            return segmented_preds
        
    def tuple_to_string(self,start_end_tuple, ndigits=1):
        return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


    def format_as_transcription(self,raw_segments):
        
        return "\n\n".join(
            [
                chunk["speaker"] + " " + self.tuple_to_string(chunk["timestamp"]) + chunk["text"]
                for chunk in raw_segments
            ]
        )
    
    def calling_fuction(self,audio_path):
        transcription = self.trascribe(audio_path)
        dairization = self.speaker_dairization(audio_path)
        labeled_speakers = self.labeling_speakers(dairization,transcription)
        final_transcription = self.format_as_transcription(labeled_speakers)
        
        return final_transcription
                
    
        
        


  
if __name__ == "__main__":
    audio_path = "audio.wav"
    inference = Inference()
    transcription = inference.calling_fuction(audio_path)
    print(transcription)
