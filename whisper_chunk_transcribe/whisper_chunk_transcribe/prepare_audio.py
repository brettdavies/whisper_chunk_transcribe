# Standard Libraries
import os
import asyncio
from pathlib import Path
from typing import Any, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

# CLI, Logging, Configuration
import fire
from loguru import logger
from dotenv import load_dotenv

# First Party Libraries
from .logger_config import LoggerConfig
from .utils import Utils
from .database import DatabaseOperations
from .process_audio_wrapper import process_audio_sync

# Load environment variables
load_dotenv()

# Configure loguru
LOGGER_NAME = "prepare_audio"
LoggerConfig.setup_logger(log_name=LOGGER_NAME, log_level=os.getenv("LOG_LEVEL", "INFO"))

class QueueManager:
    def __init__(self):
        self.source_file_path_queue = asyncio.Queue()

@dataclass(slots=True)
class PrepareAudio:
    queue_manager: QueueManager
    db_ops: DatabaseOperations
    logger: Any

    def __init__(self):
        """
        Initializes the Fetcher with a QueueManager and DatabaseOperations instance.
        """
        self.logger = logger
        self.queue_manager = QueueManager()
        self.db_ops = DatabaseOperations()

    async def worker(
            self, 
            worker_id: int, 
            model_for_vad: Path, 
            output_dir: Path, 
            executor: ProcessPoolExecutor
        ) -> None:
        """
        Worker coroutine that processes audio files from the queue.

        Args:
            worker_id (int): Identifier for the worker.
            model_for_vad (Path): Path to the model to use for VAD.
            output_dir (Path): Directory to store output files.
            executor (ProcessPoolExecutor): Executor for running CPU-bound tasks.

        Returns:
            None
        """
        worker_name = f"audio_processor_{worker_id}"

        try:
            while True:
                source_file_path: Path = await self.queue_manager.source_file_path_queue.get()
                logger.info(f"Size of video_file_queue after get: {self.queue_manager.source_file_path_queue.qsize()}")

                if source_file_path is None:
                    # Sentinel value to indicate shutdown
                    self.queue_manager.source_file_path_queue.task_done()
                    logger.warning(f"[{worker_name}] Received shutdown signal.")
                    break

                if not source_file_path.exists():
                    logger.error(f"Source file \"{source_file_path}\" does not exist")
                    self.queue_manager.source_file_path_queue.task_done()
                    continue

                try:
                    # Offload CPU-bound async task to ProcessPoolExecutor via synchronous wrapper
                    await asyncio.get_running_loop().run_in_executor(
                        executor, 
                        process_audio_sync, 
                        worker_name, 
                        source_file_path, 
                        model_for_vad, 
                        output_dir
                    )
                    logger.info(f"[{worker_name}] Processed file: {source_file_path}")
                except asyncio.CancelledError:
                    logger.error(f"[{worker_name}] Worker task was cancelled.")
                    break
                except Exception as e:
                    logger.error(f"[{worker_name}] Error processing file: {e}")
                finally:
                    self.queue_manager.source_file_path_queue.task_done()
        except asyncio.CancelledError:
            logger.error(f"[{worker_name}] Worker coroutine was cancelled.")
            raise

    async def fetch(
            self,
            output_dir: Path = None,
            model_for_vad: Path = None,
            audio_file_dir: Path = None,
            prompt_file: Path = None,
            experiment_id: int = None,
            num_workers: int = 10
        ) -> None:
        """
        Fetches and processes audio files using multiple workers.

        Args:
            output_dir (Path): Directory to store output files.
            model_for_vad (Path): Path to the model to use for VAD.
            audio_file_dir (Path): Directory containing audio files.
            prompt_file (Path): Path to the prompt file.
            experiment_id (int): Identifier for the experiment.
            num_workers (int): Number of concurrent worker tasks.

        Returns:
            None
        """
        executor = None
        workers = []
        try:
            if not isinstance(num_workers, int) or num_workers <= 0:
                logger.error(f"num_workers must be a positive integer. The passed value was: {num_workers}")
                return

            # Validate and create directories asynchronously
            output_dir: Path = await Utils.validate_and_create_dir(output_dir, "OUTPUT_DIR", True)
            if not output_dir:
                return

            model_for_vad: Path = await Utils.validate_and_create_dir(model_for_vad, "MODEL_FOR_VAD", False)
            if not model_for_vad:
                return

            audio_file_dir: Path = await Utils.validate_and_create_dir(audio_file_dir, "AUDIO_FILE_DIR", False)
            if not audio_file_dir:
                return

            # Process prompt file and insert into DB
            if prompt_file:
                try:
                    await Utils.read_prompt_terms_from_cli_argument_insert_db(prompt_file)
                except Exception as e:
                    logger.error(f"Error processing prompt terms file: {prompt_file} {e}")
                    return

            # Retrieve video local paths from the database asynchronously
            local_paths: List[str] = await asyncio.to_thread(
                self.db_ops.get_exp_video_ids, 
                experiment_id, 
                "fetch"
            )
            for local_path in local_paths:
                full_path: Path = Path(audio_file_dir) / local_path
                if not full_path.exists():
                    logger.error(f"File {full_path} does not exist")
                    continue  # Skip missing files instead of returning
                else:
                    await self.queue_manager.source_file_path_queue.put(full_path)

            logger.debug(f"Queue contents: {self.queue_manager.source_file_path_queue}")
            logger.info(f"Size of queue after put: {self.queue_manager.source_file_path_queue.qsize()}")

            # Initialize ProcessPoolExecutor
            max_workers = min(num_workers, os.cpu_count() or 1)
            executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.warning(f"Pool initialized with {max_workers} workers.")

            # Start worker tasks
            workers = [
                asyncio.create_task(
                    self.worker(
                        worker_id=i+1, 
                        model_for_vad=model_for_vad, 
                        output_dir=output_dir,
                        executor=executor
                    )
                )
                for i in range(num_workers)
            ]

            # Wait until the queue is fully processed
            await self.queue_manager.source_file_path_queue.join()

            # Stop workers by sending sentinel values
            for _ in workers:
                await self.queue_manager.source_file_path_queue.put(None)

            # Wait for workers to finish
            await asyncio.gather(*workers)

        except asyncio.CancelledError:
            logger.error("Fetch task was cancelled. Initiating shutdown.")
            raise
        finally:
            if executor:
                executor.shutdown(wait=True)
                logger.warning("ProcessPoolExecutor has been shut down.")

            # Close the database connection
            await asyncio.to_thread(self.db_ops.close)

def cmd() -> None:
    """
    Command line interface for running the Fetcher using Fire.
    """
    fire.Fire(PrepareAudio)