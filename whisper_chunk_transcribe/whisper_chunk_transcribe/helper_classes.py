import re
import math
from typing import List
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class ExpSegment:
    segment_id: int
    video_id: str
    raw_audio_path: Path
    processed_audio_path: Path
    test_case_id: int

@dataclass
class ExpTestCase:
    experiment_id: int
    test_case_id: int
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

@dataclass
class Team:
    team_id: int
    display_name: str
    abbreviation: str

@dataclass
class Player:
    player_id: int
    name: str
    team_id: int = None  # Can be set during upsert if needed

@dataclass
class Game:  # Class representing a game
    video_id: str
    espn_id: int
    home_team: Team
    away_team: Team
    home_players: List[Player] = field(default_factory=list)
    away_players: List[Player] = field(default_factory=list)

    def get_upsert_teams(self):
        return [self.home_team, self.away_team]

    def get_upsert_players(self):
        return self.home_players + self.away_players

@dataclass
class PromptData:
    prompt: str
    tokens: int
