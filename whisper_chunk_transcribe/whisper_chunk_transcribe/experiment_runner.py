# Standard Libraries
import os
import asyncio
from pathlib import Path
from typing import Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# CLI, Logging, Configuration
import fire
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
import torch
from whisper_chunk_transcribe.utils import Utils

# First Party Libraries
from .helper_classes import ExpSegment, ExpTestCase
from .logger_config import LoggerConfig
from .database import DatabaseOperations
from .transcription_wrapper import test_segment

# Load environment variables
load_dotenv()

# Configure loguru
LOGGER_NAME = "experiment_runner"
LoggerConfig.setup_logger(log_name=LOGGER_NAME, log_level=os.environ.get("LOG_LEVEL", "INFO"), log_dir=os.environ.get("LOG_DIR", "/media/bigdaddy/data/log/"))

class QueueManager:
    def __init__(self):
        self.segment_queue = asyncio.Queue()

@dataclass(slots=True)
class ExperimentRunner:
    logger: Any
    queue_manager: QueueManager
    db_ops: DatabaseOperations
    device: str
    experiment_id: int
    model_for_transcribe: Path
    test_cases: List[ExpTestCase]

    def __init__(self):
        """
        Initializes the ExperimentRunner with the logger, queue manager, database operations, and device.
        """
        self.logger = logger
        self.queue_manager = QueueManager()
        self.db_ops = DatabaseOperations()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def worker(self, worker_id: int, executor: ThreadPoolExecutor) -> None:
        """
        Worker coroutine that processes audio transcription tasks.

        Args:
            worker_id (int): Identifier for the worker.

        Returns:
            None
        """
        worker_name = f"experiment_{worker_id}"
        try:            
            while True:
                segment: ExpSegment = await self.queue_manager.segment_queue.get()

                if segment is None:
                    self.queue_manager.segment_queue.task_done()
                    logger.warning(f"[{worker_name}] Received shutdown signal.")
                    break

                # Extract a TestCase object from the list of test cases where the test_case_id matches test_case_id
                test_case: ExpTestCase = next((tc for tc in self.test_cases if tc.test_case_id == segment.test_case_id), None)

                try:
                    # Offload async task to ThreadPoolExecutor via synchronous wrapper
                    await asyncio.get_running_loop().run_in_executor(
                        executor, 
                        test_segment, 
                        worker_name,
                        self.model_for_transcribe,
                        segment,
                        test_case
                    )
                    logger.info(f"[{worker_name}] Processed segment: {segment.segment_id}")
                except asyncio.CancelledError:
                    logger.error(f"[{worker_name}] Worker task was cancelled.")
                    break
                except Exception as e:
                    logger.error(f"[{worker_name}] Error processing segment {segment.segment_id}: {e}")
                finally:
                    self.queue_manager.segment_queue.task_done()
        except Exception as e:
            logger.error(f"[{worker_name}] Error in worker: {e}")

    async def retrieve_exp_experiment_test_cases(self) -> None:
        self.test_cases = await asyncio.to_thread(
            self.db_ops.get_exp_experiment_test_cases,
            "ExperimentRunner",
            self.experiment_id
        )
        # logger.debug(f"Test case details: {str(self.test_cases)}")
        logger.info(f"Number of test cases: {len(self.test_cases)}")

    async def retrieve_exp_experiment_segments(self, use_prev_transcription: bool) -> None:
        """
        Retrieve video local paths from the database asynchronously and add them to the segment queue.

        Args:
            ref_prev_transcription (bool): Flag to determine if the segments reference previous transcriptions.

        Returns:
            None
        """
        segments: List[ExpSegment] = await asyncio.to_thread(
            self.db_ops.get_exp_experiment_segments,
            "ExperimentRunner",
            self.experiment_id,
            use_prev_transcription
        )
        for segment in segments:
            logger.debug(f"Adding segment {segment.segment_id} to the queue.")
            if not Path(segment.raw_audio_path).exists():
                logger.error(f"File {segment.raw_audio_path} does not exist")
                continue  # Skip missing files instead of returning
            elif not Path(segment.processed_audio_path).exists():
                logger.error(f"File {segment.processed_audio_path} does not exist")
                continue  # Skip missing files instead of returning
            else:
                await self.queue_manager.segment_queue.put(segment)

        # logger.debug(f"Queue contents: {self.queue_manager.segment_queue}")
        logger.info(f"Size of queue after put: {self.queue_manager.segment_queue.qsize()}")

    async def fetch(
            self,
            model_for_transcribe: Path = None,
            experiment_id: int = None,
            num_workers: int = 1
        ) -> None:
        """
        Fetch and process audio segments using multiple workers.

        Args:
            model_for_transcribe (Path): Directory containing the model to use for transcription.
            num_models (int): Number of models to load.
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

            if experiment_id and (not isinstance(experiment_id, int) or experiment_id <= 0):
                logger.error(f"experiment_id must be a positive integer. The passed value was: {experiment_id}")
                return

            if not experiment_id:
                self.experiment_id = int(os.environ.get("EXPERIMENT_ID", 0))
                if not self.experiment_id:
                    self.experiment_id = await asyncio.to_thread(
                       self.db_ops.get_most_recent_modified_exp_id,
                       "ExperimentRunner"
                    )
                    logger.warning(f"Experiment ID not set. Using most recently modified experiment ID: {self.experiment_id}")
                    if not self.experiment_id:
                        logger.error("No experiment ID found in database. Please ensure the experiment and test_cases are created in the database.")

            self.model_for_transcribe = model_for_transcribe
            if not isinstance(self.model_for_transcribe, Path):
                self.model_for_transcribe = None

            self.model_for_transcribe: Path = await Utils.validate_and_create_dir(model_for_transcribe, "MODEL_FOR_TRANSCRIBE", False)
            if not self.model_for_transcribe:
                logger.error(f"Error validating model_for_transcribe. Set in .env or pass as argument: {self.model_for_transcribe}")
                return
            else:
                logger.info(f"Model for transcription: {self.model_for_transcribe}")

            # Retrieve test cases
            await self.retrieve_exp_experiment_test_cases()

            # Retrieve segments for test cases that do NOT reference previous transcriptions
            # await self.retrieve_exp_experiment_segments(is_dynamic=False)
            await self.retrieve_exp_experiment_segments(use_prev_transcription=True)

            # Create workers
            if self.device == "cpu":
                cpu_max = 1
                try: cpu_max = int(round(os.cpu_count() * 0.80, 0))
                except: pass
                max_workers = min(num_workers, cpu_max or 1)
            else:
                max_workers = num_workers
            
            if self.queue_manager.segment_queue.qsize() > 0:
                # Initialize ThreadPoolExecutor
                executor = ThreadPoolExecutor(max_workers=max_workers)

                # Start worker tasks
                workers = [
                    asyncio.create_task(
                        self.worker(
                            worker_id=i + 1, 
                            executor=executor
                        )
                    )
                    for i in range(max_workers)
                ]

                # Wait until the queue is fully processed
                await self.queue_manager.segment_queue.join()
                logger.info("All non-dynamic segments processed.")

                # # Retrieve segments for dynamic test cases
                # await self.retrieve_exp_experiment_segments(use_prev_transcription=True)

                # # Wait until the queue is fully processed
                # await self.queue_manager.segment_queue.join()

                # Stop workers by sending sentinel values
                for _ in workers:
                    await self.queue_manager.segment_queue.put(None)

                # Wait for workers to finish
                await asyncio.gather(*workers)

        except asyncio.CancelledError:
            logger.error("Fetch task was cancelled. Initiating shutdown.")
            raise
        except Exception as e:
            logger.error(f"Error in fetch: {e}")
            raise

        finally:
            # Close the database connection
            await asyncio.to_thread(self.db_ops.close)
            # logger.warning("Database connection closed.")
            
def cmd() -> None:
    """
    Command line interface for running the Fetcher using Fire.
    """
    fire.Fire(ExperimentRunner)