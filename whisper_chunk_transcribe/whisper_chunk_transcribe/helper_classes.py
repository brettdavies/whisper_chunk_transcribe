import math
from typing import List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ExpSegment:
    segment_id: int
    raw_audio_path: Path
    processed_audio_path: Path
    test_case_id: int

@dataclass
class ExpTestCase:
    experiment_id: int
    test_case_id: int
    test_prompt_id: int
    prompt_template: str
    prompt_tokens: int
    is_dynamic: bool

@dataclass
class TranscriptionWord:
    start: float
    end: float
    word: str
    probability: float

@dataclass
class TranscriptionSegment:
    transcribed_text: str
    start: float
    end: float
    avg_logprob: float

@dataclass
class Transcription:
    segments: List[TranscriptionSegment]  # List of TranscriptionSegment instances
    words: List[TranscriptionWord]        # List of TranscriptionWord instances

def compute_average_logprob(segments: List[TranscriptionSegment]) -> float:
    """
	Converting to Probabilities and Back:
	•	More accurate for scenarios where you need to represent the average behavior of probabilities.
	•	This method is essential when you need to make probabilistic interpretations based on the average value.
    """

    # Step 1: Convert avg_logprob to probabilities
    probabilities = [math.exp(segment.avg_logprob) for segment in segments]

    if len(probabilities) == 0:
        return None

    # Step 2: Calculate the average of probabilities
    avg_probability = sum(probabilities) / len(probabilities)
    
    # Step 3: Convert the average probability back to log probability
    avg_logprob = math.log(avg_probability)
    
    return avg_logprob
