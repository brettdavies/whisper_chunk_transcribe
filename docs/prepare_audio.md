# `prepare_audio.py`

## Design Considerations and Enhancements

- **Asynchronous Task Management**: The `QueueManager` class manages the queue of audio tasks using `asyncio.Queue`, which allows for efficient, non-blocking processing of audio files in parallel. This ensures scalability when handling large volumes of audio data.
- **Multi-Process Execution**: By leveraging `ProcessPoolExecutor`, the `PrepareAudio` class offloads CPU-bound tasks (like audio processing) to separate processes, enabling concurrent execution of tasks while avoiding bottlenecks.
- **Logging**: The module uses `loguru` for comprehensive logging, providing real-time visibility into the audio preparation workflow. This ensures easier debugging and monitoring of tasks.
- **Modular and Flexible Design**: The `PrepareAudio` class is designed to be flexible, allowing for easy modification of audio processing methods and configurations via environment variables.

## Module Overview

The `prepare_audio.py` module is responsible for managing the preparation and processing of audio files. It handles asynchronous task management using `asyncio.Queue` and parallel processing using `ProcessPoolExecutor`. Logging is configured using `loguru` to ensure comprehensive tracking of all events.

### Key Libraries and Dependencies

- **`asyncio`**: Used to manage asynchronous tasks and the audio file processing queue.
  - [Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- **`concurrent.futures`**: Utilized for managing parallel execution of audio processing tasks via `ProcessPoolExecutor`.
  - [Concurrent Futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- **`fire`**: A command-line interface creation tool that simplifies the execution of scripts.
  - [Fire Documentation](https://github.com/google/python-fire)

---

## Classes and Methods

### `QueueManager`

The `QueueManager` class manages the `asyncio.Queue`, which is used to handle tasks related to processing audio files. This ensures that tasks are queued efficiently and can be processed asynchronously.

---

### `PrepareAudio`

The `PrepareAudio` class orchestrates the preparation and processing of audio files. It handles asynchronous task management, logging, and database interactions to fetch necessary data for audio processing.

#### Method: `worker(self, worker_id: int, model_for_vad: Path, output_dir: Path, executor: ProcessPoolExecutor) -> None`

The `worker()` coroutine processes audio tasks asynchronously. It retrieves audio files from the queue, processes them, and handles the audio preparation task using a model (e.g., Voice Activity Detection model). The audio processing task is offloaded to a separate process using the `ProcessPoolExecutor`.

- **How it Works**:
  - **Offloading Tasks**: The `worker()` method offloads the audio processing task (`process_audio_sync`) to a separate process using `asyncio.get_running_loop().run_in_executor()`. This enables concurrent execution of CPU-bound tasks in an otherwise asynchronous environment.
  - **Queue Management**: The method continuously retrieves tasks from an `asyncio.Queue`, processes the audio files, and marks tasks as completed.
  - **Error Handling**: Includes error handling for task cancellation and general exceptions to ensure smooth execution even if tasks fail.

```python
async def worker(self, worker_id: int, model_for_vad: Path, output_dir: Path, executor: ProcessPoolExecutor) -> None:
    """
    Worker coroutine that processes audio files from the queue asynchronously, offloading the task to a separate process using ProcessPoolExecutor.
    
    Args:
        worker_id (int): Identifier for the worker.
        model_for_vad (Path): Path to the model used for Voice Activity Detection.
        output_dir (Path): Directory where processed audio files will be saved.
        executor (ProcessPoolExecutor): Executor for offloading tasks to separate processes.

    Returns:
        None
    """
```

#### Method: `fetch(self, model_for_vad: Path, output_dir: Path, num_workers: int = 1) -> None`

The `fetch()` method orchestrates the fetching and processing of audio files using multiple worker tasks. It retrieves the list of audio files from the database and spawns workers to process the audio files concurrently using a `ProcessPoolExecutor`.

- **Asynchronous Task Offloading**: The method uses `asyncio.to_thread()` to offload database queries to a separate thread, ensuring that the main event loop is not blocked while waiting for the database response.

```python
async def fetch(self, model_for_vad: Path, output_dir: Path, num_workers: int = 1) -> None:
    """
    Fetch and process audio files using multiple workers.

    Args:
        model_for_vad (Path): Path to the model used for Voice Activity Detection.
        output_dir (Path): Directory where processed audio files will be saved.
        num_workers (int): Number of concurrent worker tasks.

    Returns:
        None
    """
```

---

## Usage Example

The following example demonstrates how to use the `PrepareAudio` class to fetch and process audio files using multiple workers:

```python
import asyncio
from pathlib import Path
from prepare_audio import PrepareAudio

# Initialize the PrepareAudio class
audio_preparer = PrepareAudio()

# Example paths for demonstration
model_path = Path("path/to/vad_model")
output_dir = Path("path/to/output")
num_workers = 4

async def run_audio_preparation():
    # Fetch and process audio files with multiple workers
    await audio_preparer.fetch(
        model_for_vad=model_path,
        output_dir=output_dir,
        num_workers=num_workers
    )

# Run the audio preparation asynchronously
asyncio.run(run_audio_preparation())
```

### Key Steps in the Example:

1. **Initialize the `PrepareAudio` Class**: This sets up logging, queue management, and database connections for audio processing.
   
2. **Fetch and Process Audio Files**:
   - The `fetch()` method retrieves the audio files from the database and processes them concurrently using multiple workers. The audio files are prepared using the provided Voice Activity Detection (VAD) model, and the output is saved in the specified directory.

This example demonstrates how to leverage the asynchronous and parallel processing features of the `PrepareAudio` class to handle large-scale audio preparation tasks efficiently.
