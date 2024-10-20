# `helper_classes.py`

## Design Considerations and Enhancements

- **Data Class Simplicity**: This module uses Python’s `dataclasses` to represent structured entities such as segments, test cases, transcriptions, and players. This approach simplifies the representation of complex data, reduces boilerplate code, and allows for easy manipulation and access to class fields.
- **Encapsulation of Experiment Data**: Classes like `ExpSegment` and `ExpTestCase` encapsulate experiment-specific data, while others like `TranscriptionWord` and `TranscriptionSegment` encapsulate transcription-related data. This structure makes it easier to manage and interact with experiment and transcription data across the system.
- **Log Integration**: By integrating `loguru`, the module allows for efficient logging to track the manipulation and status of segments, test cases, transcriptions, and other data components.

## Module Overview

The `helper_classes.py` module provides several data classes representing different components of the experiment and transcription framework, including segments, test cases, transcription words, teams, and players. Each class encapsulates specific metadata and provides utility methods where applicable.

### Key Libraries and Dependencies

- **`dataclasses`**: Provides a decorator and functions for automatically adding special methods to user-defined classes.
  - [Dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)
- **`pathlib`**: Standard library used for manipulating filesystem paths, particularly for audio file paths in this case.
  - [Pathlib Documentation](https://docs.python.org/3/library/pathlib.html)

---

## Classes and Methods

### `ExpSegment`

The `ExpSegment` class represents a segment of an experiment, encapsulating information such as the segment ID, associated video ID, and the paths to the raw and processed audio files.

```python
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
```

---

### `ExpTestCase`

The `ExpTestCase` class represents a test case for an experiment, encapsulating the experiment ID, test case ID, prompt template, and related data. Additionally, it provides property methods to manage dynamic prompt and previous transcription flags.

```python
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
```

#### Properties:
##### `is_dynamic(self) -> bool`

This property method returns the value of the `_is_dynamic` field, indicating whether the test case's prompt template is dynamic.

```python
@property
def is_dynamic(self) -> bool:
    """
    Property method to determine if the test case is dynamic.
    
    Returns:
        bool: True if the test case is dynamic, False otherwise.
    """
```

##### `use_prev_transcription(self) -> bool`

This property method returns the value of the `_use_prev_transcription` field, indicating whether the test case uses a previous transcription.

```python
@property
def use_prev_transcription(self) -> bool:
    """
    Property method to determine if the test case uses previous transcription.
    
    Returns:
        bool: True if it uses previous transcription, False otherwise.
    """
```

### `TranscriptionWord`

The `TranscriptionWord` class represents a word in a transcription, encapsulating metadata such as the word's start and end time, and the probability of its correctness.

```python
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
```

---

### `TranscriptionSegment`

The `TranscriptionSegment` class represents a segment of a transcription, storing information such as the transcribed text, start and end times, and the average log probability.

```python
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
```

---

### `Transcription`

The `Transcription` class represents a full transcription, consisting of a list of transcription segments and words.

```python
@dataclass
class Transcription:
    """
    Represents a transcription.

    Attributes:
        segments (List[TranscriptionSegment]): A list of TranscriptionSegment instances.
        words (List[TranscriptionWord]): A list of TranscriptionWord instances.
    """
```

---

### `Team`

The `Team` class represents a team, including attributes such as the team ID, display name, and abbreviation.

```python
@dataclass
class Team:
    """
    Represents a team.

    Attributes:
        team_id (int): The ID of the team.
        display_name (str): The display name of the team.
        abbreviation (str): The abbreviation of the team.
    """
```

---

### `Player`

The `Player` class represents a player, including attributes such as the player ID, name, and team ID.

```python
@dataclass
class Player:
    """
    Represents a player.

    Attributes:
        player_id (int): The ID of the player.
        name (str): The name of the player.
        team_id (int, optional): The ID of the team the player belongs to. Defaults to None and can be set during upsert if needed.
    """
```

---

### `Game`

The `Game` class represents a game, including the associated video ID, ESPN ID, teams, and lists of players.

#### Methods:
##### `get_upsert_teams(self) -> List[Team]`

Returns a list of the home and away teams involved in the game.

```python
def get_upsert_teams(self) -> List[Team]:
    """
    Returns a list of teams involved in the game.

    Returns:
        List[Team]: A list containing the home team and the away team.
    """
```

##### `get_upsert_players(self) -> List[Player]`

Returns a combined list of players from both the home and away teams for upsert operations.

```python
def get_upsert_players(self) -> List[Player]:
    """
    Returns a list of players that need to be upserted into the database.

    This method combines the home_players and away_players lists and returns the result.

    Returns:
        List[Player]: A list of players that need to be upserted into the database.
    """
```

---

### `PromptData`

The `PromptData` class represents the data for a prompt, including the text prompt and the number of tokens in it.

```python
@dataclass
class PromptData:
    """
    Represents prompt data.

    Attributes:
        prompt (str): The prompt.
        tokens (int): The number of tokens in the prompt.
    """
```

---

## Usage Example

Below is an example demonstrating how to use the `ExpSegment` and `ExpTestCase` classes to define a segment and a test case within an experiment:

```python
from pathlib import Path
from helper_classes import ExpSegment, ExpTestCase

# Define a segment of an experiment
segment = ExpSegment(
    segment_id=1,
    video_id="VID123",
    raw_audio_path=Path("/path/to/raw/audio.wav"),
    processed_audio_path=Path("/path/to/processed/audio.wav"),
    test_case_id=10
)

# Define a test case for an experiment
test_case = ExpTestCase(
    experiment_id=1,
    test_case_id=1001,
    prompt_template="Transcribe the following:",
    prompt_tokens=50,
    test_prompt_id=1234,
    _is_dynamic=True,  # Initialize the internal dynamic flag
    _use_prev_transcription=False  # Initialize the previous transcription flag
)

# Accessing segment attributes
print(f"Segment ID: {segment.segment_id}, Video ID: {segment.video_id}")
print(f"Raw Audio Path: {segment.raw_audio_path}, Processed Audio Path: {segment.processed_audio_path}")

# Accessing test case attributes
print(f"Experiment ID: {test_case.experiment_id}, Test Case ID: {test_case.test_case_id}")
print(f"Prompt Template: {test_case.prompt_template}, Tokens: {test_case.prompt_tokens}")

# Accessing property methods
print(f"Is dynamic: {test_case.is_dynamic}")
print(f"Uses previous transcription: {test_case.use_prev_transcription}")
```


Below is an example of how to use the `Transcription`, `TranscriptionSegment`, and `TranscriptionWord` classes to define transcription data and a game instance:

```python
from helper_classes import Transcription, TranscriptionSegment, and TranscriptionWord

# Define transcription segments and words
segment = TranscriptionSegment(
    transcribed_text="Hello, world!",
    start=0.0,
    end=2.0,
    avg_logprob=-1.23
)

word = TranscriptionWord(
    start=0.0,
    end=0.5,
    word="Hello",
    probability=0.95
)

# Create a transcription object
transcription = Transcription(
    segments=[segment],
    words=[word]
)

# Accessing transcription attributes
print(f"Transcribed text: {transcription.segments[0].transcribed_text}")
print(f"Word: {transcription.words[0].word}, Probability: {transcription.words[0].probability}")
```

Below is an example of how to use the `Game`, `Team`, and `Player` classes to define transcription data and a game instance:

```python
from helper_classes import Game, Team, Player
# Define teams and players
home_team = Team(team_id=1, display_name="Team A", abbreviation="TA")
away_team = Team(team_id=2, display_name="Team B", abbreviation="TB")

player1 = Player(player_id=1, name="Player 1", team_id=1)
player2 = Player(player_id=2, name="Player 2", team_id=2)

# Create a game instance
game = Game(
    video_id="VID123",
    espn_id=1001,
    home_team=home_team,
    away_team=away_team,
    home_players=[player1],
    away_players=[player2]
)

# Access team and player data
print(f"Home team: {game.home_team.display_name}, Away team: {game.away_team.display_name}")
print(f"Players: {game.get_upsert_players()}")
```
