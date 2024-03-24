from pyannote.audio import Pipeline
from pyannote.audio import Audio
import numpy as np

group_by_speaker = True
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch

io = Audio(mono="downmix", sample_rate=16000)
waveform ,sample_rate = io("audio1.wav")
pipeline.to(torch.device("cuda:0"))
print(torch.device("cuda"))


# apply pretrained pipeline
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
#diarization = pipeline("audio1.wav")
#print(diarization)

segments = []
for segment, track, label in diarization.itertracks(yield_label=True):
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


from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,use_flash_attention_2=True
)
model.to("cuda:0")

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    generate_kwargs={"language": "english","task":"transcribe"},
    chunk_length_s=30,
    batch_size=32,
    return_timestamps=True,
    torch_dtype=torch.float16,
    device="cuda:0"
)

result = pipe("audio1.wav")
transcript = result["chunks"]

# get the end timestamps for each chunk from the ASR output
end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
print(len(end_timestamps))
print(len(new_segments))
segmented_preds = []

# align the diarizer timestamps and the ASR timestamps
for segment in new_segments:
    # get the diarizer end timestamp
    end_time = segment["segment"]["end"]
    # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
    print(end_timestamps, end_time)
    upto_idx = np.argmin(np.abs(end_timestamps - end_time))

    if group_by_speaker:
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


def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(raw_segments):
    
    return "\n\n".join(
        [
            ("Doctor: " if chunk["speaker"] == "SPEAKER_00" else "Patient: ") + chunk["text"]
            for chunk in raw_segments
        ]
    )

result = format_as_transcription(segmented_preds)
