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
from .helper_classes import TranscriptionWord, TranscriptionSegment, Transcription

# Load environment variables
load_dotenv()

class TranscriptionProcessor:
    def __init__(self, worker_name: str, model: WhisperModel, db_ops: DatabaseOperations) -> None:
        self.worker_name: str = worker_name
        self.db_ops = db_ops
        self.model = model
        self.tokenizer = self.model.hf_tokenizer
        self.test_prompt_id: int = None
        self.test_prompt: str = None
        self.test_prompt_tokens: int = 0
        self.transcription: Transcription = None

    async def determine_token_length(self, prompt_to_tokenize) -> int:
        """
        Determine the length of tokens for specific words and phrases.
        """
        tokens = len(self.tokenizer.encode(prompt_to_tokenize))
        return tokens

    async def set_initial_prompt(self, test_prompt_id: int, test_prompt: str, test_prompt_tokens: int) -> None:
        self.test_prompt_id = test_prompt_id
        self.test_prompt = test_prompt
        self.test_prompt_tokens = test_prompt_tokens

    async def prepare_initial_prompt(self, prompt_template: str, experiment_test_case_id: int, is_dynamic: bool) -> None:
        """
        Prepare the initial prompt for the transcription process.

        Args:
            prompt_template (str): The template for the initial prompt.

        Returns:
            None
        """
        try:
            interim_prompt = prompt_template

            if not interim_prompt:
                self.prompt_template = None
                self.prompt_tokens = 0
            elif not is_dynamic:
                self.test_prompt = interim_prompt
                return
            
            else:
                # # Replace placeholders in the prompt template
                # # Supposrted placeholders include: {prompt_terms}, {previous_transcription}
                # interim_prompt = interim_prompt.replace("{{previous_transcription}}", str(self.previous_transcription))
                # interim_prompt = interim_prompt.replace("{{prompt_terms}}", str(self.prompt_terms))
                self.test_prompt = interim_prompt
                self.test_prompt_tokens = await self.determine_token_length(self.test_prompt)

            logger.debug(f"[{self.worker_name}] Retrieving test prompt ID: \"{self.test_prompt}\"")
            self.test_prompt_id = self.db_ops.insert_test_prompt(self.worker_name, self.test_prompt, self.test_prompt_tokens, experiment_test_case_id)

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error preparing initial prompt: {e}")

    async def transcribe_audio(self, segment_id: int, audio_path: Path) -> None:
        """
        Transcribe the raw and processed audio files for a given segment.
        The function writes the transcription, word-level confidence scores, 
        and segment-level confidence scores to separate files.

        Parameters:
        None

        Returns:
        None
        """
        try:
            # Transcribe the audio file with a prompt
            if self.test_prompt:
                logger.debug(f"[{self.worker_name}] Transcribing with initial prompt (token length: {await self.determine_token_length(self.test_prompt)}: \"{self.test_prompt}\"")
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
                raise Exception(f"No segments found for file: \"{audio_path}\"")
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
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error processing file {audio_path}: {e}")
            raise e
