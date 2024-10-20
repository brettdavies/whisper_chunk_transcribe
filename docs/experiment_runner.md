# `experiment_runner.py`

## Design Considerations and Enhancements

- **Asynchronous Task Management**: The `QueueManager` class is built around `asyncio.Queue`, allowing for efficient management of task queues in an asynchronous environment. This ensures non-blocking execution of experiment-related tasks.
- **Device Management**: The `ExperimentRunner` class dynamically assigns processing to either a GPU or CPU based on the availability of CUDA, optimizing the execution of transcription models and other compute-heavy tasks.
- **Database Integration**: The `ExperimentRunner` class interacts with `DatabaseOperations` to store and retrieve relevant data throughout the experiment lifecycle.
- **Comprehensive Logging**: Using `loguru`, the module ensures detailed tracking of all experiment-related events, simplifying the debugging and monitoring of experiments.
- **Modular Design**: The `ExperimentRunner` is designed to be flexible, allowing for easy modification of test cases, models, and transcription strategies without disrupting the core orchestration logic.

## Module Overview

The `experiment_runner.py` module handles the orchestration of experiments, including task management, transcription testing, and database operations. It uses asynchronous queues to manage the workflow efficiently and supports GPU or CPU processing for compute-heavy tasks.

### Key Libraries and Dependencies

- **`asyncio`**: Used to handle asynchronous tasks and manage the experiment queue.
  - [Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- **`torch`**: A library used for managing model inference on either the GPU or CPU.
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- **`fire`**: A command-line interface creation tool that simplifies the execution of scripts and modules.
  - [Fire Documentation](https://github.com/google/python-fire)
- **`psycopg2`**: PostgreSQL adapter used via `DatabaseOperations` for database interactions.
  - [Psycopg2 Documentation](https://www.psycopg.org/docs/)

---

## Classes and Methods

### `QueueManager`

The `QueueManager` class is responsible for managing an `asyncio.Queue` that handles the queuing of tasks in the experiment. It ensures that tasks are managed asynchronously and in a non-blocking manner.

#### Constructor: `__init__(self) -> None`

Initializes the `QueueManager` with an `asyncio.Queue` that can be used to manage tasks asynchronously.

---

### `ExperimentRunner`

The `ExperimentRunner` class orchestrates the execution of experiments, manages database interactions, and handles the queuing of tasks. It also ensures that tasks are executed on the appropriate device (GPU or CPU).

#### Constructor: `__init__(self) -> None`

Initializes the `ExperimentRunner` class with the following components:
- Logger for tracking the execution of experiments.
- `QueueManager` for handling asynchronous tasks.
- `DatabaseOperations` to interact with the database.
- A device (GPU or CPU) based on the availability of CUDA.

#### Method: `worker(self, worker_id: int, executor: ThreadPoolExecutor) -> None`

The `worker()` coroutine processes audio transcription tasks for an experiment. It continuously retrieves segments from the queue, processes them, and handles transcription using a model. The transcription task is offloaded to a separate thread using the `ThreadPoolExecutor` via `asyncio.get_running_loop().run_in_executor()`. This ensures that synchronous code (the transcription task) can be executed concurrently in an asynchronous environment without blocking the main event loop.

- **Parameters**:
  - `worker_id` (int): Identifier for the worker, used for logging purposes and tracking the worker's tasks.
  - `executor` (ThreadPoolExecutor): A thread pool executor that offloads synchronous tasks to a separate thread, allowing CPU-bound tasks to run in parallel without blocking the asynchronous event loop.

- **How it Works**:
  - **Offloading Tasks**: The `worker()` method offloads the transcription task (`test_segment`) to a separate thread by using `asyncio.get_running_loop().run_in_executor()`. This method takes the `executor`, the `test_segment` function (which is synchronous), and the necessary arguments (worker name, transcription model, audio segment, and test case). This allows the synchronous `test_segment` task to be executed in parallel, while the `worker()` method continues to manage other tasks in the event loop.
  - **Task Queue Management**: The method retrieves segments from an `asyncio.Queue`, processes them, and marks them as completed with `queue_manager.segment_queue.task_done()`.
  - **Error Handling**: It includes error handling for task cancellation (`asyncio.CancelledError`) and general exceptions, ensuring that the worker can gracefully handle interruptions or failures.

- **Returns**: None

```python
async def worker(self, worker_id: int, executor: ThreadPoolExecutor) -> None:
    """
    Worker coroutine that processes audio transcription tasks asynchronously, offloading the transcription task to a separate thread using ThreadPoolExecutor.
    
    Args:
        worker_id (int): Identifier for the worker.
        executor (ThreadPoolExecutor): Executor for offloading tasks to a separate thread.

    Returns:
        None
    """
```

#### Method: `retrieve_exp_experiment_test_cases(self) -> None`

This method retrieves the test cases for the current experiment asynchronously from the database. It offloads the database query operation to a background thread using `asyncio.to_thread()` to prevent blocking the main event loop while waiting for the database response. Once the test cases are retrieved, it logs the number of cases found.

- **Returns**: None
- **Asynchronous Task Offloading**: The method uses `asyncio.to_thread()` to offload the database query (`self.db_ops.get_exp_experiment_test_cases`) to a separate thread. This allows the main event loop to remain responsive while the database operation runs in the background.

```python
async def retrieve_exp_experiment_test_cases(self) -> None:
    """
    Retrieve experiment test cases from the database asynchronously.

    Returns:
        None
    """
```

#### Method: `retrieve_exp_experiment_segments(self, use_prev_transcription: bool) -> None`

This method asynchronously retrieves the audio segments associated with the current experiment from the database and adds them to the processing queue. It uses `asyncio.to_thread()` to offload the database query to a background thread. The method also checks if the audio files exist locally before adding the segments to the queue. If any file is missing, it logs an error and skips that segment.

- **Parameters**:
  - `use_prev_transcription` (bool): Determines if the segments should reference previous transcriptions.
  
- **Returns**: None
- **Asynchronous Task Offloading**: The method offloads the database query (`self.db_ops.get_exp_experiment_segments`) to a background thread using `asyncio.to_thread()`. This prevents blocking the event loop while waiting for the database response and allows segment processing to be queued once retrieved.

```python
async def retrieve_exp_experiment_segments(self, use_prev_transcription: bool) -> None:
    """
    Retrieve video local paths from the database asynchronously and add them to the segment queue.

    Args:
        use_prev_transcription (bool): Flag to determine if the segments reference previous transcriptions.

    Returns:
        None
    """
```

#### Method: `fetch(self, model_for_transcribe: Path = None, experiment_id: int = None, num_workers: int = 1) -> None`

This method orchestrates the fetching and processing of audio segments using multiple worker tasks. It retrieves experiment details (e.g., test cases, segments) and spawns workers to process the segments concurrently using a `ThreadPoolExecutor`. The method ensures that the correct experiment ID and transcription model are used, and it allows the number of workers to be dynamically adjusted. It also offloads certain blocking tasks (e.g., database queries) to a background thread using `asyncio.to_thread()`.

- **Parameters**:
  - `model_for_transcribe` (Path): Path to the model directory used for transcription.
  - `experiment_id` (int): The identifier for the experiment.
  - `num_workers` (int): The number of concurrent workers that will process the segments.
  
- **Returns**: None
- **Asynchronous Task Offloading**: The method offloads various database queries to a separate thread using `asyncio.to_thread()` to prevent blocking the main event loop. Additionally, it uses a `ThreadPoolExecutor` to manage worker tasks that process the segments asynchronously.

```python
async def fetch(self, model_for_transcribe: Path = None, experiment_id: int = None, num_workers: int = 1) -> None:
    """
    Fetch and process audio segments using multiple workers.

    Args:
        model_for_transcribe (Path): Directory containing the model to use for transcription.
        experiment_id (int): Identifier for the experiment.
        num_workers (int): Number of concurrent worker tasks.

    Returns:
        None
    """
```

---

## Usage Example

The following example demonstrates how to use the `ExperimentRunner` class to fetch and process audio segments using multiple workers, manage test cases, and handle asynchronous task execution.

```python
import asyncio
from pathlib import Path
from experiment_runner import ExperimentRunner

# Initialize the ExperimentRunner
runner = ExperimentRunner()

# Example values for demonstration
model_path = Path("path/to/transcription/model")
experiment_id = 1
num_workers = 4

async def run_experiment():
    # Retrieve experiment test cases asynchronously
    await runner.retrieve_exp_experiment_test_cases()

    # Retrieve experiment segments asynchronously, use previous transcriptions if available
    await runner.retrieve_exp_experiment_segments(use_prev_transcription=True)

    # Fetch and process audio segments with multiple workers
    await runner.fetch(
        model_for_transcribe=model_path,
        experiment_id=experiment_id,
        num_workers=num_workers
    )

# Run the experiment asynchronously
asyncio.run(run_experiment())
```

### Key Steps in the Example:

1. **Initialize the `ExperimentRunner`**: The `ExperimentRunner` is initialized, which sets up the necessary components such as logging, queue management, and database connections.

2. **Retrieve Experiment Test Cases**:
   - The `retrieve_exp_experiment_test_cases()` method is called to asynchronously fetch the test cases from the database using `asyncio.to_thread()`, offloading the database operation to a separate thread to avoid blocking the main event loop.

3. **Retrieve Experiment Segments**:
   - The `retrieve_exp_experiment_segments()` method is used to asynchronously fetch the audio segments associated with the experiment. The segments are retrieved from the database and added to the task queue if the corresponding audio files exist locally.

4. **Fetch and Process Segments with Workers**:
   - The `fetch()` method is called to start processing the audio segments using multiple worker tasks. It dynamically manages the number of workers and offloads synchronous tasks (such as transcription) to a `ThreadPoolExecutor` using the `worker()` method.

This example illustrates how to leverage the asynchronous and concurrent capabilities of the `ExperimentRunner` to handle large-scale experiments efficiently.