import re
import math
from typing import List
from pathlib import Path
from loguru import logger
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
    test_prompt_id: int = field(default=None, init=False)  # Set when prompt is upserted into the database.
    _is_dynamic: bool = field(default=False, init=False, repr=False)  # Internal field. Set post initialization.
    _use_prev_transcription: bool = field(default=False, init=False, repr=False)  # Internal field. Set post initialization.

    def __post_init__(self):
        """
        Post-initialization processing to set 'is_dynamic' and 'previous_transcription' based on 'prompt_template'.
        """
        if self.prompt_template:
            if not isinstance(self.prompt_template, str):
                raise ValueError(f"[ExpTestCase] prompt_template must be a string, got {type(self.prompt_template).__name__}")

            # Regular expression to find all placeholders in the format {placeholder_name}
            # Placeholder names consist of letters (case-insensitive) and digits
            placeholder_pattern = r'\{([A-Za-z0-9_]+)\}'
            detected_placeholders = re.findall(placeholder_pattern, self.prompt_template)

            # Log detected placeholders
            logger.debug(f"[ExpTestCase] Test Case {self.test_case_id}: Detected placeholders ({detected_placeholders})")

            # Set 'is_dynamic' to True if any placeholders are found
            if detected_placeholders:
                self._is_dynamic = True
                logger.debug(f"[ExpTestCase] Test Case {self.test_case_id}: `is_dynamic` set to True")

            # Check for the presence of "{previous_transcription}" in the prompt_template
            if "{previous_transcription}" in self.prompt_template:
                self._use_prev_transcription = True
                logger.debug(f"[ExpTestCase] Test Case {self.test_case_id}: `use_prev_transcription` set to True")

    @property
    def is_dynamic(self) -> bool:
        """
        Read-only property for 'is_dynamic'.
        """
        return self._is_dynamic

    @property
    def use_prev_transcription(self) -> bool:
        """
        Read-only property for 'use_prev_transcription'.
        """
        return self._use_prev_transcription

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
