# Standard Libraries
import json
from pathlib import Path
from typing import List

# CLI, Logging, Configuration
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
from faster_whisper import WhisperModel

# First Party Libraries
from .database import DatabaseOperations
from .helper_classes import TranscriptionWord, TranscriptionSegment, Transcription, ExpTestCase, ExpSegment, PromptData

# Load environment variables
load_dotenv()

class TranscriptionProcessor:
    """
    Class responsible for processing transcriptions.
        worker_name (str): The name of the worker.
        model (WhisperModel): The Whisper model.
        db_ops (DatabaseOperations): The database operations object.
    Attributes:
        worker_name (str): The name of the worker.
        db_ops (DatabaseOperations): The database operations object.
        model (WhisperModel): The Whisper model.
        tokenizer (Hugging Face Tokenizer): The tokenizer used by the model.
        test_prompt_id (int): The ID of the test prompt.
        test_prompt (str): The test prompt.
        test_prompt_tokens (int): The number of tokens in the test prompt.
        transcription (Transcription): The transcription object.
    Methods:
        determine_token_length: Determines the length of tokens for specific words and phrases.
        set_initial_prompt: Sets the initial prompt for the transcription process.
        build_prompt: Builds the initial prompt by adding words from the end of the full transcription.
        prepare_initial_prompt: Prepares the initial prompt for the transcription process.
        transcribe_audio: Transcribes the audio file for a given segment.
        Initializes a new instance of the TranscriptionProcessor class.
            worker_name (str): The name of the worker.
            model (WhisperModel): The Whisper model.
            db_ops (DatabaseOperations): The database operations object.
        Determines the length of tokens for specific words and phrases.
            prompt_to_tokenize: The prompt to tokenize.
            int: The number of tokens in the prompt.
        ...
        Sets the initial prompt for the transcription process.
            test_prompt_id (int): The ID of the test prompt.
            test_prompt (str): The test prompt.
            test_prompt_tokens (int): The number of tokens in the test prompt.
        ...
        Builds the initial prompt by adding words from the end of the full transcription.
            initial_prompt (str): The template prompt containing the placeholder {previous_transcription}.
            full_transcription (str): The complete transcription text.
            max_tokens (int): The maximum allowed number of tokens.
            initial_step (int): Number of words to add initially.
            step_decrement (int): Number to decrement the step by when the token limit is exceeded.
            str: The constructed prompt with previous_transcription inserted.
        ...
        Prepares the initial prompt for the transcription process.
            segment (ExpSegment): The segment object.
        ...
        Transcribes the audio file for a given segment.
            segment_id (int): The ID of the segment.
            audio_path (Path): The path to the audio file.
        """
    def __init__(self, worker_name: str, model: WhisperModel, db_ops: DatabaseOperations) -> None:
        """
        Initializes a TranscriptionProcessor object.

        Args:
            worker_name (str): The name of the worker.
            model (WhisperModel): The WhisperModel object used for transcription.
            db_ops (DatabaseOperations): The DatabaseOperations object used for database operations.

        Returns:
            None
        """
        self.worker_name: str = worker_name
        self.db_ops = db_ops
        self.model = model
        self.tokenizer = self.model.hf_tokenizer
        self.test_prompt_id: int = None
        self.test_prompt: str = None
        self.test_prompt_tokens: int = 0
        self.transcription: Transcription = None

    async def determine_token_length(self, prompt_to_tokenize: str) -> int:
        """
        Determines the length of tokens for specific words and phrases.

        Args:
            prompt_to_tokenize (str): The prompt to tokenize.

        Returns:
            int: The number of tokens in the prompt.
        """
        tokens = len(self.tokenizer.encode(prompt_to_tokenize))
        return tokens

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
        self.test_prompt_id = test_prompt_id
        self.test_prompt = test_prompt
        self.test_prompt_tokens = test_prompt_tokens

    async def build_prompt(self, initial_prompt: str, full_transcription: str, max_tokens: int = 255, initial_step: int = 10, step_decrement: int = 1) -> str:
        """
        Builds the initial prompt by adding words from the end of the full transcription.
        Starts by adding 'initial_step' words at a time.
        If adding 'initial_step' words exceeds 'max_tokens', decrement the step by 'step_decrement' and try again.
        Continues until the token limit is reached without exceeding it.

        Args:
            initial_prompt (str): The template prompt containing the placeholder {previous_transcription}.
            full_transcription (str): The complete transcription text.
            max_tokens (int, optional): The maximum allowed number of tokens. Defaults to 255.
            initial_step (int, optional): Number of words to add initially. Defaults to 10.
            step_decrement (int, optional): Number to decrement the step by when the token limit is exceeded. Defaults to 1.

        Returns:
            str: The constructed prompt with {previous_transcription} inserted.
        """
        words = full_transcription.split()
        total_words = len(words)

        best_fit_transcription = ""
        current_step = initial_step
        start = total_words  # Start from the end
        end = total_words

        while start > 0 and current_step > 0:
            # Move the start pointer back by 'current_step' words
            start = max(start - current_step, 0)
            subset_words = words[start:end]
            subset_transcription = ' '.join(subset_words)

            # Replace the placeholder with the current subset
            prompt = initial_prompt.replace("{previous_transcription}", subset_transcription.strip() + " " + best_fit_transcription.strip())

            # Determine the token length
            tokens = await self.determine_token_length(prompt)

            if tokens > max_tokens:
                logger.debug(f"[{self.worker_name}] Tokens {tokens}\n{prompt}")
                if current_step == 1:
                    # Cannot add more words without exceeding the limit
                    break
                else:
                    # Reset the starting point
                    start = max(start + current_step, 0)
                    # Reduce the step size and try again
                    current_step -= step_decrement
                    logger.debug(f"[{self.worker_name}] {tokens} exceeds token limit. Decrementing step to {current_step}")
            else:
                # Update the best fit and expand the window
                best_fit_transcription = subset_transcription.strip() + " " + best_fit_transcription.strip()
                logger.debug(f"[{self.worker_name}] Tokens {tokens} Updated best fit transcription:\n{best_fit_transcription}")
                
                # Reset the step to initial_step for the next iteration
                current_step = initial_step
                
                # Continue to add more words
                end = start

        # Final prompt with the best fit transcription
        final_prompt = initial_prompt.replace("{previous_transcription}", best_fit_transcription.strip())
        return final_prompt

    async def prepare_initial_prompt(self, test_case: ExpTestCase, segment: ExpSegment) -> List[PromptData]:
        """
        Prepare the initial prompt for the transcription process.

        Args:
            test_case (ExpTestCase): The test case object.
            segment (ExpSegment): The segment object.

        Returns:
            List[PromptData]: A list of PromptData objects containing the initial prompt and token length.
        """
        try:
            # Initialize variables
            initial_prompt = test_case.prompt_template
            raw_initial_prompt = test_case.prompt_template
            processed_initial_prompt = test_case.prompt_template

            # Build initial_prompt with teams and players
            if "{teamA}" in test_case.prompt_template or \
                "{teamB}" in test_case.prompt_template or \
                "{playerA}" in test_case.prompt_template or \
                "{playerB}" in test_case.prompt_template:
                # Retrieve teams and players from the database
                teams, players = [] , []
                teams_players = self.db_ops.get_teams_players(self.worker_name, segment.video_id)
                if teams_players:
                    for team in teams_players:
                        teams.append(team[0])
                        players.append(team[1])
                    initial_prompt = f"{initial_prompt.replace("{teamA}", teams[0]).replace("{teamB}", teams[1]).replace("{playerA}", players[0]).replace("{playerB}", players[1])}"
                    logger.debug(f"[{self.worker_name}] Interim prompt: {initial_prompt}")

            # Build initial_prompt with previous transcription
            if test_case.use_prev_transcription:
                # Retrieve the previous transcription from the database
                transcripts = self.db_ops.get_previous_transcription(self.worker_name, test_case.experiment_id, segment.segment_id)
                if transcripts:
                    processed_transcription = transcripts[0][1]
                    logger.debug(f"[{self.worker_name}] Processed transcription:\n{processed_transcription}")
                    raw_transcription = transcripts[1][1]
                    logger.debug(f"[{self.worker_name}] Raw transcription:\n{raw_transcription}")

                    # Build raw_initial_prompt
                    raw_initial_prompt = await self.build_prompt(initial_prompt = initial_prompt, full_transcription = raw_transcription)
                    # # Build processed_initial_prompt
                    processed_initial_prompt = await self.build_prompt(initial_prompt = initial_prompt, full_transcription = processed_transcription)

            if test_case.is_dynamic:
                raw_tokens = await self.determine_token_length(raw_initial_prompt)
                processed_tokens = await self.determine_token_length(processed_initial_prompt)
                logger.debug(f"[{self.worker_name}] Raw initial prompt ({raw_tokens}):\n{raw_initial_prompt}")
                logger.debug(f"[{self.worker_name}] Processed initial prompt ({processed_tokens}):\n{processed_initial_prompt}")

                return PromptData(prompt=raw_initial_prompt, tokens=raw_tokens), PromptData(prompt=processed_initial_prompt, tokens=processed_tokens)

            else:
                tokens = await self.determine_token_length(initial_prompt)

                return PromptData(prompt=initial_prompt, tokens=tokens)

        except ValueError as e:
            logger.error(f"[{self.worker_name}] ValueError preparing initial prompt: {e}")
            raise e
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error preparing initial prompt: {e}")
            raise e

    async def transcribe_audio(self, audio_path: Path) -> None:
        """
        Transcribes the audio file for a given segment.

        Args:
            segment_id (int): The ID of the segment.
            audio_path (Path): The path to the audio file.

        Returns:
            None
        """
        try:
            # Transcribe the audio file with a prompt
            if self.test_prompt:
                logger.debug(f"[{self.worker_name}] Transcribing with initial prompt")
                segments_generator, info = self.model.transcribe(str(audio_path), beam_size=5, word_timestamps=True, initial_prompt=self.test_prompt)
            
            # Transcribe the audio file without a prompt
            else:
                logger.debug(f"[{self.worker_name}] Transcribing without initial prompt")
                segments_generator, info = self.model.transcribe(str(audio_path), beam_size=5, word_timestamps=True)

            logger.debug(f"[{self.worker_name}] Transcription Info:\n{info}")

            # Convert generator to list
            segments = list(segments_generator)
            if not segments:
                logger.error(f"[{self.worker_name}] No segments found for file: \"{audio_path}\"")
                raise Exception((f"No segments found for file: \"{audio_path}\""))
            else:
                logger.debug(f"[{self.worker_name}] Number of segments: {len(segments)}")
                for segment in segments:
                    logger.debug(f"[{self.worker_name}] {segment}")

            # Prepare to store results
            transcription_segments = []
            transcription_words = []

            # Extract transcription and confidence scores
            for segment in segments:
                # Create a TranscriptionSegment instance
                transcription_segment = TranscriptionSegment(
                    transcribed_text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    avg_logprob=segment.avg_logprob,
                )

                # Append to the transcription_segments list
                transcription_segments.append(transcription_segment)

                # Check if words attribute is present and not None
                if segment.words:
                    for word in segment.words:
                        transcription_word = TranscriptionWord(
                            start=word.start,
                            end=word.end,
                            word=word.word,
                            probability=word.probability
                        )
                        transcription_words.append(transcription_word)
                else:
                    logger.warning(f"[{self.worker_name}] No word-level details for segment: {segment.text}")

            # Create a Transcription instance
            self.transcription = Transcription(segments=transcription_segments, words=transcription_words)

            # # Write the transcription text to a file
            # transcription_file = audio_path.replace(".wav", ".txt")
            # with open(transcription_file, 'w') as f:
            #     f.write("\n".join([segment.transcribed_text for segment in transcription_segments]))
            # logger.debug(f"[{self.worker_name}] Transcription written to: \"{transcription_file}\"")

            # # Write word-level confidence scores to a file
            # word_confidence_file = audio_path.replace(".wav", "_word_confidences.json")
            # with open(word_confidence_file, 'w') as f:
            #     json.dump([word.__dict__ for word in transcription_words], f, indent=4)
            # logger.debug(f"[{self.worker_name}] Word-level confidence scores written to: \"{word_confidence_file}\"")

            # # Write segment-level confidence scores to a file
            # segment_confidence_file = audio_path.replace(".wav", "_segment_confidences.json")
            # segment_confidences = [
            #     {
            #         "start": segment.start,
            #         "end": segment.end,
            #         "avg_logprob": segment.avg_logprob,
            #     }
            #     for segment in transcription_segments
            #     ]
            # with open(segment_confidence_file, 'w') as f:
            #     json.dump(segment_confidences, f, indent=4)
            # logger.debug(f"[{self.worker_name}] Segment-level confidence scores written to: \"{segment_confidence_file}\"")

        except ValueError as e:
            logger.error(f"[{self.worker_name}] ValueError processing file {audio_path}: {e}")
            raise e
        except FileNotFoundError as e:
            logger.error(f"[{self.worker_name}] File not found error processing file {audio_path}: {e}")
            raise e
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error processing file {audio_path}: {e}")
            raise e
