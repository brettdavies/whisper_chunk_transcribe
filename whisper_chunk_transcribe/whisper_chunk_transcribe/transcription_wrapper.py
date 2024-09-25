# Standard Libraries
import asyncio
from pathlib import Path

# CLI, Logging, Configuration
from loguru import logger

# Third Party Libraries
from faster_whisper import WhisperModel
import torch

# First Party Libraries
from .helper_classes import ExpSegment, ExpTestCase, compute_average_logprob
from .database import DatabaseOperations
from .transcription_processor import TranscriptionProcessor

def test_segment(worker_name: str, model_for_transcribe: Path, segment: ExpSegment, test_case: ExpTestCase) -> None:
    """
    Synchronous wrapper to run the asynchronous tests.

    Args:
        worker_name (str): Identifier for the worker.
        device (str): Device to run the model on.
        model_for_transcribe (Path): Path to the model for transcription.
        segment (ExpSegment): Instance of the ExpSegment class.
        prompt_template (str): Template for the prompt.

    Returns:
        None
    """

    async def runner() -> None:
        db_ops = DatabaseOperations()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            async def load_model(device: str, model_for_transcribe: Path) -> WhisperModel:
                """
                Load an instance of the model into the device.

                Args:
                    device (str): Device on which to to load the model.
                    model_for_transcribe (Path): Path to the model for transcription.

                Returns:
                    WhisperModel: An instance of the WhisperModel.
                """
                model = None
                try:
                    model = WhisperModel(model_size_or_path=str(model_for_transcribe), device=device)
                    logger.info(f"[{worker_name}] Loaded model into {device}.")
                except Exception as e:
                    logger.error(f"[{worker_name}] Error loading model: {e}")
                    raise e

                return model

            async def set_test_results(worker_name: str, db_ops: DatabaseOperations, test_case: ExpTestCase, segment: ExpSegment, transcription_processor: TranscriptionProcessor, is_raw_audio: bool) -> None:
                """
                Insert the results of the experiment test into the database.

                Args:
                    worker_name (str): Identifier for the worker.
                    db_ops (DatabaseOperations): Instance of the DatabaseOperations class.
                    test_case (ExpTestCase): Instance of the ExpTestCase class.
                    segment (ExpSegment): Instance of the ExpSegment class.
                    transcription_processor (TranscriptionProcessor): Instance of the TranscriptionProcessor class.

                Returns:
                    None
                """

                # Calculate average segment average log probability
                average_segment_avg_logprob = compute_average_logprob(transcription_processor.transcription.segments)

                # Calculate average word probability
                if len(transcription_processor.transcription.words) == 0:
                    average_word_probability = None
                average_word_probability = sum([word.probability for word in transcription_processor.transcription.words]) / len(transcription_processor.transcription.words)

                # Insert experiment test
                test_id = db_ops.insert_exp_test(
                    worker_name,
                    test_case,
                    segment.segment_id,
                    is_raw_audio,
                    average_word_probability,
                    average_segment_avg_logprob
                )

                # Insert experiment test transcriptions
                transcription = ' '.join(segment.transcribed_text for segment in transcription_processor.transcription.segments)
                db_ops.insert_exp_test_transcriptions(worker_name, test_id, transcription)

                # Insert experiment test transcription segments
                db_ops.insert_exp_test_transcription_segments(worker_name, test_id, transcription_processor.transcription.segments)

                # Insert experiment test transcription words
                db_ops.insert_exp_test_transcription_words(worker_name, test_id, transcription_processor.transcription.words)

            # Load the model
            model = await load_model(device, model_for_transcribe)

            transcription_processor = TranscriptionProcessor(worker_name, model, db_ops)
            
            if not test_case.is_dynamic and test_case.prompt_tokens:
                await transcription_processor.set_initial_prompt(test_case.test_case_id, test_case.prompt_template, test_case.prompt_tokens)
            # else:
            #     # Prepare the initial prompt
            #     logger.debug(f"[{worker_name}] Preparing initial prompt: {test_case.prompt_template}")
            #     await transcription_processor.prepare_initial_prompt(test_case.prompt_template, test_case.test_case_id, test_case.is_dynamic)

            # Call the transcription method for the raw audio file
            logger.debug(f"[{worker_name}] Performing transcription for raw: \"{segment.raw_audio_path}\"")
            await transcription_processor.transcribe_audio(segment.segment_id, segment.raw_audio_path)

            # Set the test results for the raw audio file
            await set_test_results(worker_name, db_ops, test_case, segment, transcription_processor, is_raw_audio=True)

            # Reset the transcription object
            transcription_processor.transcription = None

            # Call the transcription method for the processed audio file
            logger.debug(f"[{worker_name}] Performing transcription for processed:\"{segment.processed_audio_path}\"")
            await transcription_processor.transcribe_audio(segment.segment_id, segment.processed_audio_path)

            # Set the test results for the processed audio file
            await set_test_results(worker_name, db_ops, test_case, segment, transcription_processor, is_raw_audio=False)

        except Exception as e:
            logger.error(f"[{worker_name}] Error in test_segment: {e}")
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(runner())
        logger.debug(f"[{worker_name}] test_segment completed for {segment.segment_id}")
    finally:
        loop.close()
