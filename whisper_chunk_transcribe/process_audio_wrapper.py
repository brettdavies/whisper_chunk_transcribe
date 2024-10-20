# Standard Libraries
import asyncio
from pathlib import Path

# CLI, Logging, Configuration
from loguru import logger

# First Party Libraries
from .audio_processor import AudioProcessor
from .database import DatabaseOperations

def process_audio_sync(worker_name: str, source_file_path: Path, model_for_vad: Path, output_dir: Path) -> None:
    """
    Synchronous wrapper to run the asynchronous process_audio method.
    
    Args:
        worker_name (str): Identifier for the worker.
        source_file_path (Path): Path to the source audio file to be processed.
        model_for_vad (Path): Path to the Voice Activity Detection (VAD) model used for segmenting the audio.
        output_dir (Path): Directory where processed audio files and segments will be saved.
    
    Returns:
        None
    """
    async def runner() -> None:
        """
        Asynchronous function that processes audio files.

        This function performs the following steps:
        1. Transforms the source file to WAV format.
        2. Creates Voice Activity Detection (VAD) segments from the audio file.
        3. Segments the audio based on VAD.
        4. Calculates Signal-to-Noise Ratio (SNR) for each segment.

        Raises:
            Exception: If an error occurs during the audio processing.

        Returns:
            None
        """
        db_ops = DatabaseOperations()
        try:
            audio_processor = AudioProcessor(model_for_vad, source_file_path, output_dir, worker_name, db_ops)

            # Transform source file to WAV
            logger.debug(f"[{worker_name}] Transforming \"{source_file_path}\"")
            await audio_processor.transform_source2wav()

            # Create VAD segments from the audio file
            logger.debug(f"[{worker_name}] VAD Processing \"{source_file_path}\"")
            await audio_processor.extract_segments_from_audio_file()

            # Segment the audio based on VAD
            logger.debug(f"[{worker_name}] Creating segments \"{source_file_path}\"")
            await audio_processor.segment_audio_based_on_vad()

            # Calculate SNR for each segment
            logger.debug(f"[{worker_name}] Calculating SNR on segments")
            await audio_processor.calculate_snr()

        except Exception as e:
            logger.error(f"[{worker_name}] Error in process_audio_async: {e}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(runner())
        logger.debug(f"[{worker_name}] process_audio_sync completed for {source_file_path}")
    finally:
        loop.close()