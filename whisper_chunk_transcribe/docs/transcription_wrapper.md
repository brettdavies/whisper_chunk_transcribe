# `transcription_wrapper.py`

## Design Considerations and Enhancements

- **Synchronous Wrapper for Asynchronous Transcription Tasks**: The `test_segment` function provides a synchronous wrapper for running the asynchronous transcription tests, allowing easier integration in environments that may not natively support asynchronous operations.
- **Device Management**: The transcription process dynamically selects the available device, either GPU (`cuda`) or CPU, for model loading and execution, optimizing performance.
- **Detailed Transcription Pipeline**: This module supports both raw and processed audio transcriptions, including dynamic prompt preparation, transcription execution, and detailed result logging and storage.
- **Comprehensive Logging and Error Handling**: The module uses `loguru` for detailed logging at every stage of the transcription and result storage process. It ensures robust error handling and logging of any issues that arise during model loading, transcription, or database interactions.

## Module Overview

The `transcription_wrapper.py` module is a synchronous wrapper that handles the transcription of audio segments. It manages the entire transcription workflow, including model loading, dynamic prompt preparation, and saving the results into the database. This module supports asynchronous execution internally but exposes a synchronous interface through the `test_segment` function.

### Key Libraries and Dependencies

- **`asyncio`**: Handles asynchronous tasks such as model loading, transcription, and database operations while providing synchronous control through an event loop.
  - [Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- **`faster_whisper`**: A library used to transcribe audio using the Whisper model.
  - [Faster Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- **`torch`**: Used to check device availability (GPU/CPU) and load the Whisper model appropriately.
  - [Torch Documentation](https://pytorch.org/docs/stable/index.html)

---

## Functions

### `test_segment`

The `test_segment` function is a synchronous wrapper that runs the transcription workflow asynchronously. It transcribes both raw and processed audio files using the Whisper model and stores the results in the database.

#### Function: `test_segment(worker_name: str, model_for_transcribe: Path, segment: ExpSegment, test_case: ExpTestCase) -> None`

This function synchronizes the entire transcription workflow, from loading the model and transcribing audio segments to storing the results in the database. It handles both raw and processed audio files, managing the transcription for each file separately.

- **How it Works**:
  - **Model Loading**: The model is loaded onto the appropriate device (GPU or CPU) based on availability.
  - **Dynamic Prompt Handling**: The function prepares prompts dynamically based on the test case. If required, previous transcriptions are used to build a contextually relevant prompt.
  - **Raw and Processed Audio**: The function transcribes both raw and processed versions of the audio file and logs the results in the database.
  - **Database Operations**: After transcription, the results, including segment and word-level transcriptions, are inserted into the database.

```python
def test_segment(worker_name: str, model_for_transcribe: Path, segment: ExpSegment, test_case: ExpTestCase) -> None:
    """
    This function synchronizes the entire transcription workflow,
    from loading the model and transcribing audio segments to storing the results in the database.
    It handles both raw and processed audio files, managing the transcription for each file separately.

    Args:
        worker_name (str): Identifier for the worker.
        model_for_transcribe (Path): Path to the model for transcription.
        segment (ExpSegment): Instance of the ExpSegment class.
        test_case (ExpTestCase): Instance of the ExpTestCase class.

    Returns:
        None
    """
```

#### Inner Function: `load_model(device: str, model_for_transcribe: Path) -> WhisperModel`

Loads the Whisper model onto the specified device (GPU or CPU) and logs the device used.

```python
async def load_model(device: str, model_for_transcribe: Path) -> WhisperModel:
    """
    Load an instance of the model into the device.

    Args:
        device (str): Device on which to load the model.
        model_for_transcribe (Path): Path to the model for transcription.

    Returns:
        WhisperModel: An instance of the WhisperModel.
    """
```

#### Inner Function: `set_test_results(worker_name: str, db_ops: DatabaseOperations, test_case: ExpTestCase, segment: ExpSegment, transcription_processor: TranscriptionProcessor, is_raw_audio: bool) -> None`

Inserts the results of the transcription test into the database, including the average log probability for each segment and word, as well as the transcription text.

```python
async def set_test_results(worker_name: str, db_ops: DatabaseOperations, test_case: ExpTestCase, segment: ExpSegment, transcription_processor: TranscriptionProcessor, is_raw_audio: bool) -> None:
    """
    Insert the results of the experiment test into the database.

    Args:
        worker_name (str): Identifier for the worker.
        db_ops (DatabaseOperations): Database operations instance for storing results.
        test_case (ExpTestCase): The current test case configuration.
        segment (ExpSegment): The audio segment being transcribed.
        transcription_processor (TranscriptionProcessor): The processor handling transcription tasks.
        is_raw_audio (bool): Flag indicating if the audio is raw or processed.

    Returns:
        None
    """
```

---

## Usage Example

The following example demonstrates how to use the `test_segment` function to transcribe an audio segment using the Whisper model.

```python
from pathlib import Path
from transcription_wrapper import test_segment
from helper_classes import ExpSegment, ExpTestCase

# Example values for demonstration
worker_name = "worker_1"
model_path = Path("path/to/whisper/model")
segment = ExpSegment(video_id=123, segment_id=456, raw_audio_path="path/to/raw/audio", processed_audio_path="path/to/processed/audio")
test_case = ExpTestCase(test_case_id=1, prompt_template="Prompt text", is_dynamic=False, use_prev_transcription=False)

# Run the synchronous transcription test
test_segment(worker_name, model_path, segment, test_case)
```
