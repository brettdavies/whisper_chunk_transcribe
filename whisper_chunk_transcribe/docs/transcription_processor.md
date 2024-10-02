# `transcription_processor.py`

## Design Considerations and Enhancements

- **Transcription Workflow Automation**: The `TranscriptionProcessor` class manages the transcription workflow, including audio file processing, prompt preparation, and transcription generation. It provides a seamless integration with the Whisper model for transcribing audio files with or without prompts.
- **Dynamic Prompt Handling**: The module offers dynamic prompt generation by building prompts based on previous transcriptions, making the transcription process more contextually aware. It also supports test cases that require dynamic prompts based on specific tokens.
- **Comprehensive Logging**: Detailed logs are generated for each step of the transcription process, including prompt preparation, tokenization, and transcription execution. This ensures transparency and simplifies debugging.

## Module Overview

The `transcription_processor.py` module handles the transcription of audio files using a Whisper model. It supports dynamic prompt preparation, integrates with a database to retrieve relevant data, and calculates token lengths for transcription accuracy. The module allows for the extraction of transcription segments and word-level details.

### Key Libraries and Dependencies

- **`faster_whisper`**: A library used to perform transcription tasks using the Whisper model.
  - [Faster Whisper Documentation](https://github.com/guillaumekln/faster-whisper)

---

## Classes and Methods

### `TranscriptionProcessor`

The `TranscriptionProcessor` class is responsible for processing audio files and generating transcriptions. It prepares dynamic prompts, processes audio segments, and calculates token lengths for handling prompts efficiently.

#### Constructor: `__init__(self, worker_name: str, model: WhisperModel, db_ops: DatabaseOperations) -> None`

Initializes the `TranscriptionProcessor` class with the following components:
- **Worker Name**: Identifier for the worker, used for logging.
- **Whisper Model**: The Whisper model used for audio transcription.
- **Database Operations**: Provides access to the database to retrieve test case data and previous transcriptions.

#### Method: `determine_token_length(self, prompt_to_tokenize: str) -> int`

Determines the length of tokens for a given prompt using the Hugging Face tokenizer from the Whisper model.

```python
async def determine_token_length(self, prompt_to_tokenize: str) -> int:
    """
    Determines the length of tokens for specific words and phrases.

    Args:
        prompt_to_tokenize (str): The prompt to tokenize.

    Returns:
        int: The number of tokens in the prompt.
    """
```

#### Method: `set_initial_prompt(self, test_prompt_id: int, test_prompt: str, test_prompt_tokens: int) -> None`

Sets the initial transcription prompt using the test case data.

```python
async def set_initial_prompt(self, test_prompt_id: int, test_prompt: str, test_prompt_tokens: int) -> None:
    """
    Sets the initial prompt for the transcription processor.

    Args:
        test_prompt_id (int): The ID of the test prompt.
        test_prompt (str): The text of the test prompt.
        test_prompt_tokens (int): The number of tokens in the test prompt.

    Returns:
        None
    """
```

#### Method: `build_prompt(self, initial_prompt: str, full_transcription: str, max_tokens: int = 255, initial_step: int = 10, step_decrement: int = 1) -> str`

Builds the transcription prompt by adding words from the end of a full transcription. The function adjusts the number of words added to fit within a specified token limit.

```python
async def build_prompt(self, initial_prompt: str, full_transcription: str, max_tokens: int = 255, initial_step: int = 10, step_decrement: int = 1) -> str:
    """
    Builds the initial prompt by adding words from the end of the full transcription.
    Starts by adding 'initial_step' words at a time.
    If adding 'initial_step' words exceeds 'max_tokens', decrement the step by 'step_decrement' and try again.

    Args:
        initial_prompt (str): The template prompt containing the placeholder {previous_transcription}.
        full_transcription (str): The complete transcription text.
        max_tokens (int, optional): The maximum allowed number of tokens. Defaults to 255.
        initial_step (int, optional): Number of words to add initially. Defaults to 10.
        step_decrement (int, optional): Number to decrement the step by when the token limit is exceeded. Defaults to 1.

    Returns:
        str: The constructed prompt with {previous_transcription} inserted.
    """
```

#### Method: `prepare_initial_prompt(self, test_case: ExpTestCase, segment: ExpSegment) -> List[PromptData]`

Prepares the initial prompt for the transcription process. It handles dynamic prompts by incorporating previous transcriptions and relevant metadata from the database (e.g., teams or players).

```python
async def prepare_initial_prompt(self, test_case: ExpTestCase, segment: ExpSegment) -> List[PromptData]:
    """
    Prepares the initial prompt for the transcription process.
    It handles dynamic prompts by incorporating previous transcriptions
    and relevant metadata from the database (e.g., teams or players).

    Args:
        test_case (ExpTestCase): The test case object.
        segment (ExpSegment): The segment object.

    Returns:
        List[PromptData]: A list of PromptData objects containing the initial prompt and token length.
    """
```

#### Method: `transcribe_audio(self, audio_path: Path) -> None`

Transcribes the audio file for a given segment. It can either use a prompt or transcribe without one, depending on the test case settings. The method captures the transcribed segments and words, including their timestamps and confidence scores.

```python
async def transcribe_audio(self, audio_path: Path) -> None:
    """
    Transcribes the audio file for a given segment.
    It can either use a prompt or transcribe without one, depending on the test case settings.
    The method captures the transcribed segments and words, including their timestamps and confidence scores.

    Args:
        audio_path (Path): The path to the audio file.

    Returns:
        None
    """
```

---

## Usage Example

The following example demonstrates how to use the `TranscriptionProcessor` class to transcribe an audio file using a Whisper model:

```python
from pathlib import Path
from transcription_processor import TranscriptionProcessor
from faster_whisper import WhisperModel
from database import DatabaseOperations

# Initialize the TranscriptionProcessor
whisper_model = WhisperModel("path/to/model")
db_ops = DatabaseOperations()
transcription_processor = TranscriptionProcessor(worker_name="worker_1", model=whisper_model, db_ops=db_ops)

# Example audio file path
audio_path = Path("path/to/audio/file.wav")

async def transcribe():
    # Transcribe the audio file
    await transcription_processor.transcribe_audio(audio_path)

# Run the transcription asynchronously
import asyncio
asyncio.run(transcribe())
```
