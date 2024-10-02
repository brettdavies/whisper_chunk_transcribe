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

    async def load_model(device: str, model_for_transcribe: Path) -> WhisperModel:
        """
        Load an instance of the model into the device.

        Args:
            device (str): Device on which the model is to be loaded (`cpu` or `cuda`).
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
            db_ops (DatabaseOperations): Database operations instance for storing results.
            test_case (ExpTestCase): The current test case configuration.
            segment (ExpSegment): The audio segment being transcribed.
            transcription_processor (TranscriptionProcessor): The processor handling transcription tasks.
            is_raw_audio (bool): Flag indicating if the audio is raw or processed.

        Returns:
            None
        """

        # Calculate average segment average log probability
        average_segment_avg_logprob = compute_average_logprob(transcription_processor.transcription.segments)

        # Calculate average word probability
        average_word_probability = None
        if len(transcription_processor.transcription.words) > 0:
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

    async def runner() -> None:
        db_ops = DatabaseOperations()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.debug(f"[{worker_name}] Starting test_segment for video {segment.video_id}, segment {segment.segment_id}")
            # Load the model
            model = await load_model(device, model_for_transcribe)

            transcription_processor = TranscriptionProcessor(worker_name, model, db_ops)
            
            logger.debug(f"[{worker_name}] Test case: {test_case.test_case_id}, prompt: {test_case.prompt_template}, is_dynamic: {test_case.is_dynamic}, use_prev_transcription: {test_case.use_prev_transcription}")

            if test_case.is_dynamic:
                tokens = 0
                prompt = ""
                logger.debug(f"[{worker_name}] Preparing dynamic initial prompt: {test_case.prompt_template}")

                if test_case.use_prev_transcription:
                    logger.debug(f"[{worker_name}] Preparing dynamic initial prompt with previous transcriptions")
                    raw_data, processed_data = await transcription_processor.prepare_initial_prompt(test_case, segment)

                    logger.debug(f"[{worker_name}] Retrieving Raw test prompt ID")
                    test_case.test_prompt_id = db_ops.insert_test_prompt(worker_name, raw_data.prompt, raw_data.tokens, test_case.test_case_id)

                    logger.debug(f"[{worker_name}] Retrieving Processed test prompt ID")
                    processed_test_prompt_id = db_ops.insert_test_prompt(worker_name, processed_data.prompt, processed_data.tokens, test_case.test_case_id)

                else:
                    prompt, tokens = await transcription_processor.prepare_initial_prompt(test_case, segment)
                
                    logger.debug(f"[{worker_name}] Retrieving test prompt ID")
                    test_case.test_prompt_id = db_ops.insert_test_prompt(worker_name, prompt, tokens, test_case.test_case_id)
                
                await transcription_processor.set_initial_prompt(test_case.test_prompt_id, raw_data.prompt, raw_data.tokens)

            else:
                logger.debug(f"[{worker_name}] Retrieving test prompt ID")
                test_case.test_prompt_id = db_ops.insert_test_prompt(worker_name, test_case.prompt_template, test_case.prompt_tokens, test_case.test_case_id)
                
                await transcription_processor.set_initial_prompt(test_case.test_prompt_id, test_case.prompt_template, test_case.prompt_tokens)


            ##################
            # Raw Audio File
            ##################
            # Call the transcription method for the raw audio file
            logger.debug(f"[{worker_name}] Performing transcription for raw: \"{segment.raw_audio_path}\"")
            await transcription_processor.transcribe_audio(segment.raw_audio_path)

            # Set the test results for the raw audio file
            await set_test_results(worker_name, db_ops, test_case, segment, transcription_processor, is_raw_audio=True)


            ######################
            # Reset the transcription object
            ######################
            transcription_processor.transcription = None


            ######################
            # Processed Audio File
            ######################
            # Set the initial prompt for the processed previous transcription test case
            if test_case.use_prev_transcription:
                await transcription_processor.set_initial_prompt(processed_test_prompt_id, processed_data.prompt, processed_data.tokens)

            logger.debug(f"[{worker_name}] Performing transcription for processed:\"{segment.processed_audio_path}\"")
            await transcription_processor.transcribe_audio(segment.processed_audio_path)

            # Set the test results for the processed audio file
            await set_test_results(worker_name, db_ops, test_case, segment, transcription_processor, is_raw_audio=False)

        except Exception as e:
            logger.error(f"[{worker_name}] Error in test_segment: {e}")
        finally:
            del model
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(runner())
        logger.debug(f"[{worker_name}] test_segment completed for {segment.segment_id}")
    finally:
        loop.close()
