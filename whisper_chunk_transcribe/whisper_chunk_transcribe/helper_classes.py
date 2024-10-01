import re
import math
from typing import List
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

@dataclass
class ExpSegment:
    """
    Represents a segment of an experiment.

    Attributes:
        segment_id (int): The ID of the segment.
        video_id (str): The ID of the video associated with the segment.
        raw_audio_path (Path): The path to the raw audio file of the segment.
        processed_audio_path (Path): The path to the processed audio file of the segment.
        test_case_id (int): The ID of the test case associated with the segment.
    """
    segment_id: int
    video_id: str
    raw_audio_path: Path
    processed_audio_path: Path
    test_case_id: int

@dataclass
class ExpTestCase:
    """
    Represents a test case for an experiment.

    Attributes:
        experiment_id (int): The ID of the experiment.
        test_case_id (int): The ID of the test case.
        prompt_template (str): The template for the prompt.
        prompt_tokens (int): The number of tokens in the prompt.
        test_prompt_id (int, optional): The ID of the test prompt in the database. Defaults to None.
        _is_dynamic (bool): Internal field to indicate if the prompt template is dynamic. Defaults to False.
        _use_prev_transcription (bool): Internal field to indicate if the prompt template uses previous transcription. Defaults to False.
    """
    experiment_id: int
    test_case_id: int
    prompt_template: str
    prompt_tokens: int
    test_prompt_id: int = field(default=None, init=False)  # Set when prompt is upserted into the database.
    _is_dynamic: bool = field(default=False, init=False, repr=False)  # Internal field. Set post initialization.
    _use_prev_transcription: bool = field(default=False, init=False, repr=False)  # Internal field. Set post initialization.

    def __post_init__(self):
        """
        Post-initialization processing to set 'is_dynamic' and 'use_prev_transcription' based on 'prompt_template'.

        Raises:
            ValueError: If 'prompt_template' is not a string.
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
    """
    Represents a word in a transcription.

    Attributes:
        start (float): The start time of the word.
        end (float): The end time of the word.
        word (str): The word itself.
        probability (float): The probability of the word being correct.
    """
    start: float
    end: float
    word: str
    probability: float

@dataclass
class TranscriptionSegment:
    """
    Represents a segment of a transcription.

    Attributes:
        transcribed_text (str): The transcribed text of the segment.
        start (float): The start time of the segment.
        end (float): The end time of the segment.
        avg_logprob (float): The average log probability of the segment.
    """
    transcribed_text: str
    start: float
    end: float
    avg_logprob: float

@dataclass
class Transcription:
    """
    Represents a transcription.

    Attributes:
        segments (List[TranscriptionSegment]): A list of TranscriptionSegment instances.
        words (List[TranscriptionWord]): A list of TranscriptionWord instances.
    """
    segments: List[TranscriptionSegment]
    words: List[TranscriptionWord]

def compute_average_logprob(segments: List[TranscriptionSegment]) -> float:
    """
    Compute the average log probability of a list of TranscriptionSegments.

    Args:
        segments (List[TranscriptionSegment]): A list of TranscriptionSegment instances.

    Returns:
        float: The average log probability.

    Raises:
        TypeError: If segments is not a list.
        ValueError: If segments is an empty list.

    Notes:
        This method converts the average log probability to probabilities, calculates the average of the probabilities,
        and then converts the average probability back to log probability.
    """
    if not isinstance(segments, list):
        raise TypeError("segments must be a list")

    if len(segments) == 0:
        raise ValueError("segments cannot be an empty list")

    # Convert avg_logprob to probabilities
    probabilities = [math.exp(segment.avg_logprob) for segment in segments]

    # Calculate the average of probabilities
    avg_probability = sum(probabilities) / len(probabilities)

    # Convert the average probability back to log probability
    avg_logprob = math.log(avg_probability)

    return avg_logprob

@dataclass
class Team:
    """
    Represents a team.

    Attributes:
        team_id (int): The ID of the team.
        display_name (str): The display name of the team.
        abbreviation (str): The abbreviation of the team.
    """
    team_id: int
    display_name: str
    abbreviation: str

@dataclass
class Player:
    """
    Represents a player.

    Attributes:
        player_id (int): The ID of the player.
        name (str): The name of the player.
        team_id (int, optional): The ID of the team the player belongs to. Defaults to None and can be set during upsert if needed.
    """
    player_id: int
    name: str
    team_id: int = None

@dataclass
class Game:
    """
    Represents a game.

    Attributes:
        video_id (str): The ID of the video associated with the game.
        espn_id (int): The ID of the game in ESPN.
        home_team (Team): The home team of the game.
        away_team (Team): The away team of the game.
        home_players (List[Player], optional): The list of home players. Defaults to an empty list.
        away_players (List[Player], optional): The list of away players. Defaults to an empty list.
    """
    video_id: str
    espn_id: int
    home_team: Team
    away_team: Team
    home_players: List[Player] = field(default_factory=list)
    away_players: List[Player] = field(default_factory=list)

    def get_upsert_teams(self) -> List[Team]:
        """
        Returns a list of teams involved in the game.

        Returns:
            List[Team]: A list containing the home team and the away team.
        """
        return [self.home_team, self.away_team]

    def get_upsert_players(self) -> List[Player]:
        """
        Returns a list of players that need to be upserted into the database.

        This method combines the home_players and away_players lists and returns the result.

        Returns:
            List[Player]: A list of players that need to be upserted into the database.
        """
        return self.home_players + self.away_players

@dataclass
class PromptData:
    """
    Represents prompt data.

    Attributes:
        prompt (str): The prompt.
        tokens (int): The number of tokens in the prompt.
    """
    prompt: str
    tokens: int
