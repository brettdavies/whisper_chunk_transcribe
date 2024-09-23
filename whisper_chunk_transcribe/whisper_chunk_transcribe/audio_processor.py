# Standard Libraries
import json
import random
import asyncio
from pathlib import Path

# CLI, Logging, Configuration
from loguru import logger

# Third Party Libraries
import torch
import numpy as np
import pandas as pd
import noisereduce as nr
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.signal import wiener
from pyannote.core import Annotation
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from faster_whisper import WhisperModel

# First Party Libraries
from .utils import Utils
from .metadata import Metadata
from .database import DatabaseOperations

class AudioProcessor:
    def __init__(self, model_dir: Path, source_file_path: Path, output_dir: Path, worker_name: str, db_ops: DatabaseOperations) -> None:
        self.db_ops = db_ops
        self.worker_name = worker_name
        logger.debug(f"[{self.worker_name}] Worker: {self.worker_name}")

        self.model_dir = model_dir
        logger.debug(f"[{self.worker_name}] Model Dir: {self.model_dir}")

        self.output_dir = output_dir
        logger.debug(f"[{self.worker_name}] Output Dir: {self.output_dir}")

        self.source_file_path = source_file_path
        logger.debug(f"[{self.worker_name}] Source File Path: {self.source_file_path}")

        # Set file name and extension from the source file path
        self.source_file_name = self.source_file_path.name
        logger.debug(f"[{self.worker_name}] Source File Name: {self.source_file_name}")

        self.file_name_noext = self.source_file_path.stem
        self.source_file_extension = self.source_file_path.suffix
        logger.debug(f"[{self.worker_name}] File Name No EXT: {self.file_name_noext}")
        logger.debug(f"[{self.worker_name}] Source File Ext: {self.source_file_extension}")

        # Extract video_id
        self.video_id, _ = Utils.extract_video_info_filepath(self.file_name_noext)
        logger.debug(f"[{self.worker_name}] File Video ID: {self.video_id}")

        self.output_file_extension = ".wav"
        logger.debug(f"[{self.worker_name}] Output File Ext: {self.output_file_extension}")

        # Construct output file names and paths based on the provided directories and filenames
        self.output_file_name = self.file_name_noext + self.output_file_extension
        logger.debug(f"[{self.worker_name}] Output File Name: {self.output_file_name}")

        self.output_file_dir = self.output_dir / self.video_id
        logger.debug(f"[{self.worker_name}] Output File Dir: {self.output_file_dir}")

        self.output_original_file_dir = self.output_file_dir / "original"
        logger.debug(f"[{self.worker_name}] Output Original File Dir: {self.output_original_file_dir}")

        self.output_processed_file_dir = self.output_file_dir / "processed"
        logger.debug(f"[{self.worker_name}] Output Processed File Dir: {self.output_processed_file_dir}")

        self.original_segment_file_path = self.output_original_file_dir / "segments"
        logger.debug(f"[{self.worker_name}] Output Original Segments File Dir: {self.original_segment_file_path}")

        self.processed_segment_file_path = self.output_processed_file_dir / "segments"
        logger.debug(f"[{self.worker_name}] Output Processed Segments File Dir: {self.processed_segment_file_path}")

        # Construct full output paths
        self.output_file_original_path_name = self.output_original_file_dir / self.output_file_name
        logger.debug(f"[{self.worker_name}] Output File Original Path Name: {self.output_file_original_path_name}")

        self.output_file_processed_path_name = self.output_processed_file_dir / self.output_file_name
        logger.debug(f"[{self.worker_name}] Output File Processed Path Name: {self.output_file_processed_path_name}")

        # Set device for torch based on CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"[{self.worker_name}] Is CUDA Device Available? {self.device == 'cuda'}")

        # self.torch_dtype = "float16" if torch.cuda.is_available() else "float32"
        # logger.debug(f"[{self.worker_name}] Using PyTorch dtype: {self.torch_dtype}")

    async def transform_source2wav(self) -> None:
        """
        Complete audio processing pipeline:
        1. Load and resample audio to 16 kHz.
        2. Apply Wiener filtering.
        3. Apply Non-stationary Noise Reduction.
        4. Save the processed audio as a 16 kHz WAV file.
        """
        try:
            if self.output_file_original_path_name.exists() and self.output_file_processed_path_name.exists():
                logger.info(f"[{self.worker_name}] Original and Processed WAV files already exist for video_id \"{self.video_id}\". Not recreating them.")
            else:
                logger.info(f"[{self.worker_name}] Transforming \"{self.video_id}\" to WAV format")
                ## Create the first WAV file with load_and_resample_audio and save_waveform_to_wav
                # Load and resample the audio
                waveform, sample_rate = await self.load_and_resample_audio()
                await self.save_waveform_to_wav(waveform=waveform, sample_rate=sample_rate, is_original=True)
                
                ## Create the second WAV file with load_and_resample_audio, apply_wiener_filter, apply_non_stationary_noise_reduction, and save_waveform_to_wav
                # Apply Wiener filter to the waveform
                cleaned_waveform = await self.apply_wiener_filter(waveform)
                
                # Apply Non-stationary Noise Reduction to the waveform
                reduced_noise_waveform = await self.apply_non_stationary_noise_reduction(cleaned_waveform, sample_rate)
                
                # Save the processed waveform to a WAV file
                await self.save_waveform_to_wav(waveform=reduced_noise_waveform, sample_rate=sample_rate, is_original=False)
            
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error in transform_source2wav: {e}")

    async def load_vad_pipeline(self):
        """
        Load the Voice Activity Detection (VAD) pipeline.

        Returns:
        - pipeline (VoiceActivityDetection): The loaded VAD pipeline.
        """
        try:
            # Path to the model directory
            vad_model_dir = f"{self.model_dir}/pyannote_segmentation-3.0/pytorch_model.bin"
            model = Model.from_pretrained(
                vad_model_dir,
                repo_type="local"  # Indicates usage of a local model repository
            )

            logger.debug(f"[{self.worker_name}] Loading VAD pipeline from directory: {vad_model_dir}")

            # Initialize the VAD pipeline
            pipeline = VoiceActivityDetection(
                segmentation=model,  # Directory containing the model
                device=torch.device(self.device),  # Device (GPU or CPU)
            )

            logger.debug(f"[{self.worker_name}] VAD pipeline loaded successfully.")
            return pipeline

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error loading VAD pipeline: {e}")
            return False

    async def extract_segments_from_audio_file(self) -> None:
        """
        Process the given WAV file and segment it using voice activity detection (VAD).

        Parameters:
        None
        
        Returns:
        - Annotation
            An Annotation object containing the segments of detected speech.
        """
        try:
            # Generate the output file path for the VAD results
            output_vad_path = self.output_file_dir / (self.file_name_noext + ".txt")

            # Check if VAD results already exist
            if output_vad_path.exists():
                output_vad_path.unlink()
                logger.info(f"[{self.worker_name}] Deleted existing VAD results for \"{self.video_id}\"")

            logger.info(f"[{self.worker_name}] Generating segments for \"{self.video_id}\"")
            # Load the .wav file
            audio = AudioSegment.from_file(file=self.output_file_processed_path_name, format=self.output_file_extension)

            # Resample the audio to 16 kHz (16000 Hz) if not already
            audio = audio.set_frame_rate(16000)

            # Convert the audio to a NumPy array
            waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # Normalize the waveform to the range [-1, 1]
            if np.max(np.abs(waveform)) > 0:
                waveform /= np.max(np.abs(waveform))

            # Add a channel dimension if the waveform is mono
            if audio.channels == 1:
                waveform = waveform[np.newaxis, :]  # Add channel dimension
            
            # Convert the waveform to a PyTorch tensor and move it to the GPU
            waveform_tensor = torch.tensor(waveform).to(self.device)

            # Load the VAD pipeline
            pipeline = await self.load_vad_pipeline()
            if pipeline is None:
                logger.error(f"[{self.worker_name}] VAD pipeline is not loaded.")
                return False

            # Define hyperparameters for VAD
            HYPER_PARAMETERS = {
                "min_duration_on": 0.2,   # Remove speech regions shorter than this many seconds
                "min_duration_off": 0.2,  # Fill non-speech regions shorter than this many seconds
            }
            pipeline.instantiate(HYPER_PARAMETERS)

            # Perform VAD on the waveform tensor
            self.vad: Annotation = pipeline({"waveform": waveform_tensor, "sample_rate": 16000})
            if self.vad:
                logger.debug(f"[{self.worker_name}] VAD results:\n{self.vad}")

                with open(output_vad_path, 'w') as f:
                    f.write(str(self.vad))
                logger.info(f"[{self.worker_name}] Processed VAD for \"{self.video_id}\", found {len(self.vad)} segments.")
        
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error processing audio file into segments: {e}")
            return False

    async def update_segment_df(self, segment_df: pd.DataFrame, segment_number: int, start_time, end_time, segment_file_path: str, is_original: bool) -> pd.DataFrame:
        """
        Updates the segment_df DataFrame by adding a new row or updating an existing one based on self.video_id and segment_number.
        Handles relative file paths by prepending the execution path if the file is not found.

        Parameters:
        - segment_df (pd.DataFrame): The DataFrame to update.
        - segment_number (int): The segment number.
        - start_time: The start time of the segment.
        - end_time: The end time of the segment.
        - segment_file_path (str): Path to the audio file.
        - is_original (bool): Flag indicating if the audio is original.

        Returns:
        - pd.DataFrame: The updated DataFrame.
        """
        try:
            # Check if the specific video_id and segment_number exists
            exists = ((segment_df["video_id"] == self.video_id) & (segment_df["segment_number"] == segment_number)).any()
            if not exists:
                # Create a new row as a DataFrame
                new_row = pd.DataFrame([{
                    "video_id": self.video_id,
                    "segment_number": segment_number,
                    "start_time": start_time,
                    "end_time": end_time,
                    "raw_audio_path": segment_file_path.as_posix() if is_original else None,
                    "processed_audio_path": segment_file_path.as_posix() if not is_original else None
                }])
                # Concatenate the new row to the existing DataFrame
                segment_df = pd.concat([segment_df, new_row], ignore_index=True)
                logger.debug(f"[{self.worker_name}] Added new segment for video_id: \"{self.video_id}\", segment_number: {segment_number}")
            else:
                # Find the index of the existing row
                filtered_df = segment_df[(segment_df["video_id"] == self.video_id) & (segment_df["segment_number"] == segment_number)]
                if not filtered_df.empty:
                    segment_df_idx = filtered_df.index[0]
                    if is_original:
                        segment_df.at[segment_df_idx, "raw_audio_path"] = segment_file_path.as_posix()
                        logger.debug(f"[{self.worker_name}] Updated raw_audio_path for video_id: \"{self.video_id}\", segment_number: {segment_number}")
                    else:
                        segment_df.at[segment_df_idx, "processed_audio_path"] = segment_file_path.as_posix()
                        logger.debug(f"[{self.worker_name}] Updated processed_audio_path for video_id: \"{self.video_id}\", segment_number: {segment_number}")
                else:
                    # This should not happen as 'exists' is True
                    logger.error(f"[{self.worker_name}] Failed to locate existing segment for video_id: \"{self.video_id}\", segment_number: {segment_number}")
                    raise ValueError(f"[{self.worker_name}] Segment not found for video_id: \"{self.video_id}\", segment_number: {segment_number}")

            return segment_df

        except KeyError as key_error:
            logger.error(f"[{self.worker_name}] Key error during segmenting: {key_error}")
            raise key_error
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error during segmenting: {e}")
            raise e

    async def segment_audio_based_on_vad(self) -> None:
        """
        Segments the audio file based on VAD (Voice Activity Detection) results.
        Each VAD segment is saved as a separate audio file in a subfolder called 'segments'.

        Parameters:
        None

        Returns:
        - segment_file_path: str
            The filepath where the segments were saved.
        """
        logger.info(f"[{self.worker_name}] Segmenting audio based on VAD results for \"{self.video_id}\"")
        
        if not isinstance(self.vad, Annotation):
            raise ValueError(f"[{self.worker_name}] vad must be an Annotation object")
        try:
            # Define the input audio files
            audio_files = [
                {"file_path": self.output_file_original_path_name, "is_original": True},
                {"file_path": self.output_file_processed_path_name, "is_original": False}
            ]

            # Sample DataFrame with required columns
            self.segment_df = pd.DataFrame(columns=[
                "video_id",
                "segment_number",
                "start_time",
                "end_time",
                "raw_audio_path",
                "processed_audio_path"
            ])

            for audio_file in audio_files:
                file_path = audio_file["file_path"]
                is_original = audio_file["is_original"]
                segment_dir = None

                if is_original:
                    segment_dir = self.original_segment_file_path
                else:
                    segment_dir = self.processed_segment_file_path

                # Check if there are any segment files already created
                first_segment_path = segment_dir / "00001.wav"
                logger.debug(f"[{self.worker_name}] Checking if there are any segment files already created: \"{first_segment_path}\"")
                if first_segment_path.exists():
                    # Delete the existing segments
                    for segment_file in segment_dir.glob("*.wav"):
                        segment_file.unlink()
                    logger.info(f"[{self.worker_name}] Deleted existing segments for \"{self.output_file_name}\"")
                
                # Create a directory for the segments
                segment_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"[{self.worker_name}] Segments directory: \"{segment_dir}\"")

                # Load the audio file
                audio = AudioSegment.from_file(file_path, format=self.output_file_extension.lstrip('.'))

                segment_number = 0
                # Iterate over the segments in the VAD results and create audio segments
                for idx, segment in enumerate(self.vad.itersegments()):
                    segment_number += 1
                    start_time = segment.start
                    end_time = segment.end

                    # Convert start_time and end_time from seconds to milliseconds
                    start_ms = start_time * 1000
                    end_ms = end_time * 1000

                    # Extract the audio segment
                    segment = audio[start_ms:end_ms]

                    # Generate the segment file name
                    segment_file_name = f"{idx + 1:05}.wav"
                    segment_file_path = segment_dir / segment_file_name

                    # Export the segment as a WAV file
                    segment.export(segment_file_path, format=self.output_file_extension.lstrip('.'))

                    logger.debug(f"[{self.worker_name}] Segment \"{segment_file_name}\" saved at \"{segment_file_path}\"")

                    # Update the segment DataFrame
                    self.segment_df = await self.update_segment_df(self.segment_df, segment_number, start_time, end_time, segment_file_path, is_original)
            
            # Insert the segment DataFrame into the database
            if not self.segment_df.empty:
                logger.debug(f"[{self.worker_name}]\n{self.segment_df.to_string()}")
                await asyncio.to_thread(self.db_ops.insert_audio_segment_df, self.segment_df, self.worker_name)

            else:
                logger.debug(f"[{self.worker_name}] Segment DataFrame is empty")

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error during segmenting: {e}")
            raise ValueError(f"[{self.worker_name}] Segment file path is invalid")

    async def load_and_resample_audio(self, target_sample_rate=16000) -> tuple[np.ndarray, int]:
        """
        Load the source audio file and resample it to the target sample rate.
        
        Parameters:
        - target_sample_rate (int): The desired sample rate for the resampled audio. Default is 16000 (16kHz).
        
        Returns:
        - waveform (np.ndarray): The resampled audio waveform as a numpy array.
        - target_sample_rate (int): The actual sample rate of the resampled audio.
        """

        try:
            logger.info(f"[{self.worker_name}] Loading and resampling audio from \"{self.video_id}\" to {target_sample_rate} Hz")
            # Load the audio file using PyDub
            audio = AudioSegment.from_file(self.source_file_path, format=self.source_file_extension.lstrip('.'))
            audio = audio.set_frame_rate(target_sample_rate)
            audio = audio.set_channels(1)  # Ensure the audio is mono
            waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Optional: Normalize the waveform to the range [-1, 1]
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            # Debug: Log information about the waveform
            logger.debug(f"[{self.worker_name}] Waveform dtype: {waveform.dtype}")
            logger.debug(f"[{self.worker_name}] Waveform shape: {waveform.shape}")
            logger.debug(f"[{self.worker_name}] Waveform max value: {np.max(waveform)}")
            logger.debug(f"[{self.worker_name}] Waveform min value: {np.min(waveform)}")
            
            return waveform, target_sample_rate

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error loading and resampling audio: {e}")
            return False

    async def apply_wiener_filter(self, waveform: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Apply a Wiener filter to the input waveform.

        Parameters:
        - waveform: np.ndarray
            The input waveform to which the Wiener filter will be applied.
        - epsilon: float, optional
            A small value added to the cleaned waveform to avoid division by zero.

        Returns:
        - np.ndarray
            The waveform after applying the Wiener filter.

        Note:
        - The Wiener filter is used to enhance the quality of a signal by reducing noise.
        - If the cleaned waveform is all zeros, a warning message will be logged.
        """
        logger.info(f"[{self.worker_name}] Applying Wiener filter to the waveform")
        if not isinstance(waveform, np.ndarray):
            raise ValueError(f"[{self.worker_name}] waveform must be a numpy.ndarray")
        if not isinstance(epsilon, float):
            raise ValueError(f"[{self.worker_name}] epsilon must be a float")

        try:
            cleaned_waveform = wiener(waveform) + epsilon  # Avoid division by zero
            if np.all(cleaned_waveform == 0):
                logger.warning(f"[{self.worker_name}] Wiener filter output is all zeros.")
            return cleaned_waveform
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error applying Wiener filter: {e}")
            return waveform

    async def apply_non_stationary_noise_reduction(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply a non-stationary noise reduction filter using the noisereduce library.

        Parameters:
        - waveform: np.ndarray
            The input waveform to apply noise reduction on.
        - sample_rate: int
            The sample rate of the waveform.

        Returns:
        - np.ndarray
            The waveform after applying the non-stationary noise reduction filter.
        """
        logger.info(f"[{self.worker_name}] Applying non-stationary noise reduction to the waveform")
        if not isinstance(waveform, np.ndarray):
            raise ValueError(f"[{self.worker_name}] waveform must be a numpy.ndarray")
        if not isinstance(sample_rate, int):
            raise ValueError(f"[{self.worker_name}] sample_rate must be an int")

        try:
            cleaned_waveform = nr.reduce_noise(y=waveform, sr=sample_rate)
            return cleaned_waveform
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error applying non-stationary noise reduction: {e}")
            return waveform

    async def save_waveform_to_wav(self, waveform: np.ndarray, sample_rate: int, is_original: bool, amplification_factor: float = 2.0) -> None:
        """
        Save the waveform as a WAV file with the specified sample rate using scipy.io.wavfile.
        
        Parameters:
        - waveform: np.ndarray
            The waveform data to be saved as a WAV file.
        - sample_rate: int
            The sample rate of the waveform.
        - amplification_factor: float, optional
            The amplification factor to apply to the waveform. Default is 2.0.
        """

        out_dir: str = None
        out_filepath: str = None

        if is_original:
            out_dir = self.output_original_file_dir
            out_filepath = self.output_file_original_path_name
        else:
            out_dir = self.output_processed_file_dir
            out_filepath = self.output_file_processed_path_name

        logger.debug(f"[{self.worker_name}] Saving waveform to WAV file: \"{out_dir}\"")
        if not isinstance(waveform, np.ndarray):
            raise ValueError(f"[{self.worker_name}] waveform must be a numpy.ndarray")

        # Handle NaNs and infinities
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply amplification to the waveform
        waveform = waveform * amplification_factor
        
        # Ensure waveform is within the valid range [-1, 1] after amplification
        waveform = np.clip(waveform, -1.0, 1.0)
        
        # Convert the waveform back to 16-bit PCM for saving
        waveform_int16 = np.int16(waveform * 32767)  # Ensure scaling before converting
        
        # Save to WAV using scipy.io.wavfile
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            wavfile.write(out_filepath, sample_rate, waveform_int16)
            logger.debug(f"[{self.worker_name}] Saved WAV file to \"{out_dir}\"")
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error saving WAV file: {e}")
            return False

    async def load_model(self) -> None:
        """
        Load the Whisper model and initialize the tokenizer.

        Parameters:
        None

        Returns:
        None
        """
        try:
            # Initialize the Whisper model with the Whisper model in the local directory and the specified device (GPU)
            self.model = WhisperModel(self.model_dir, device=self.device)

            # Access the tokenizer
            self.tokenizer = self.model.hf_tokenizer

            logger.info(f"[{self.worker_name}] Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"[{self.worker_name}] Error loading Whisper model: {e}")

    async def transcribe_audio(self, is_original: bool) -> None:
        """
        Transcribe all WAV files in the given directory using the faster-whisper model.
        The function writes the transcription, word-level confidence scores, 
        and segment-level confidence scores to separate files.

        Parameters:
        None

        Returns:
        None
        """
        try:
            if is_original:
                segment_path = self.original_segment_file_path
            else:
                segment_path = self.output_file_processed_path_name

            logger.info(f"[{self.worker_name}] Transcribing audio files in directory: \"{self.vad}\"")
            
            if not self.model:
                self.load_model()

            prompt_id, initial_prompt = None

            try:
                # retrieve a random prompt from the metadata whisper_prompts list
                whisper_prompts = Metadata.whisper_prompts
                random_prompt = random.choice(whisper_prompts)
                prompt_id = random_prompt["key"]
                initial_prompt = random_prompt["value"]
            except:
                logger.error(f"[{self.worker_name}] Error retrieving random prompt from metadata")

            if segment_path.endswith(".wav"):
                file_path = self.segment_file_path / segment_path
                logger.info(f"[{self.worker_name}] Processing file: \"{file_path}\"")

                try:
                    # Transcribe the audio file with a prompt
                    if initial_prompt:
                        segments_generator, info = self.model.transcribe(file_path, beam_size=5, word_timestamps=True, initial_prompt=initial_prompt)
                    
                    # Transcribe the audio file without a prompt
                    else:
                        segments_generator, info = self.model.transcribe(file_path, beam_size=5, word_timestamps=True)

                    logger.debug(f"[{self.worker_name}] Info:\n{info}")

                    # Convert generator to list
                    segments = list(segments_generator)
                    logger.debug(f"[{self.worker_name}] Segments:\n{segments}")

                    if not segments:
                        logger.error(f"[{self.worker_name}] No segments found for file: \"{file_path}\"")

                    # Prepare to store results
                    transcription_text = []
                    word_confidences = []
                    segment_confidences = []

                    # Extract transcription and confidence scores
                    for segment in segments:
                        if segment is None:
                            continue

                        transcription_text.append(segment.text)

                        # Check if words attribute is None
                        if segment.words:
                            word_confidences.extend(segment.words)
                        else:
                            logger.warning(f"[{self.worker_name}] No word-level details for segment: {segment.text}")

                        segment_confidences.append({
                            "start": segment.start,
                            "end": segment.end,
                            "complement_avg_logprob": (1 + segment.avg_logprob)
                        })

                    # Write the transcription text to a file
                    transcription_file = file_path.replace(".wav", ".txt")
                    with open(transcription_file, 'w') as f:
                        f.write("\n".join(transcription_text))
                    logger.debug(f"[{self.worker_name}] Transcription written to: \"{transcription_file}\"")

                    # Write word-level confidence scores to a file
                    word_confidence_file = file_path.replace(".wav", "_word_confidences.json")
                    with open(word_confidence_file, 'w') as f:
                        json.dump([wc._asdict() for wc in word_confidences], f, indent=4)
                    logger.debug(f"[{self.worker_name}] Word-level confidence scores written to: \"{word_confidence_file}\"")

                    # Write segment-level confidence scores to a file
                    segment_confidence_file = file_path.replace(".wav", "_segment_confidences.json")
                    with open(segment_confidence_file, 'w') as f:
                        json.dump(segment_confidences, f, indent=4)
                    logger.debug(f"[{self.worker_name}] Segment-level confidence scores written to: \"{segment_confidence_file}\"")

                except Exception as e:
                    logger.error(f"[{self.worker_name}] Error processing file {file_path}: {e}")

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error during transcription: {e}")

    async def calculate_snr(self) -> None:
        """
        Calculate Signal-to-Noise Ratio (SNR) for each record in segment_df and update the database in batches.

        Returns:
            None
        """
        try:
            # Define batch size
            batch_size = 500
            total_records = len(self.segment_df)

            # Check if there are any segments to process
            if total_records == 0:
                logger.warning(f"[{self.worker_name}] No audio segments found to calculate SNR.")
                return

            # Process the segments in batches
            for start in range(0, total_records, batch_size):
                end = min(start + batch_size, total_records)
                batch_df = self.segment_df.iloc[start:end]

                logger.debug(f"[{self.worker_name}] Calculating SNR for {len(batch_df)} audio segments.")

                # Prepare to gather SNR values for the current batch
                snr_tasks = []
                for index, segment in batch_df.iterrows():  # Iterate over the DataFrame rows
                    raw_path = segment['raw_audio_path']
                    processed_path = segment['processed_audio_path']
                    
                    # Log the paths being used for SNR calculation
                    logger.debug(f"[{self.worker_name}] Computing SNR for raw: {raw_path}, processed: {processed_path}")
                    
                    snr_tasks.append(self.compute_snr(raw_path, processed_path))

                # Gather SNR values for the current batch
                snr_values = await asyncio.gather(*snr_tasks)

                # Add SNR values to the DataFrame
                batch_df['snr'] = snr_values

                # Update SNR values in the database for the current batch
                await asyncio.to_thread(self.db_ops.insert_audio_segment_snr, batch_df, self.worker_name)

                logger.info(f"[{self.worker_name}] Calculated SNR for {len(batch_df)} audio segments.")

        except Exception as e:
            logger.error(f"[{self.worker_name}] Error calculating SNR: {e}")

    @staticmethod
    async def compute_snr(raw_audio_path: Path, processed_audio_path: Path) -> float:
        """
        Compute the Signal-to-Noise Ratio (SNR) between raw and processed audio.

        Args:
            raw_audio_path (str): Path to the raw audio file.
            processed_audio_path (str): Path to the processed audio file.

        Returns:
            float: Calculated SNR value.    
        """

        # Load the raw and processed audio files
        rate_raw, raw_audio = wavfile.read(raw_audio_path)
        rate_processed, processed_audio = wavfile.read(processed_audio_path)

        # Ensure that both audio files have the same length
        min_length = min(len(raw_audio), len(processed_audio))
        raw_audio = raw_audio[:min_length]
        processed_audio = processed_audio[:min_length]

        # Function to calculate the power of a signal
        def calculate_power(signal):
            return np.sum(signal ** 2) / len(signal)

        # Function to calculate SNR
        def calculate_snr(clean_signal, noisy_signal):
            noise = noisy_signal - clean_signal
            signal_power = calculate_power(clean_signal)
            noise_power = calculate_power(noise)

            # Handle division by zero by returning NaN
            if noise_power == 0 or signal_power == 0:
                logger.warning("Calculated SNR: Returning NaN due to zero power.")
                return float('nan')

            snr = 10 * np.log10(signal_power / noise_power)

            return snr

        # Calculate SNR
        snr_value = calculate_snr(processed_audio, raw_audio)
        
        logger.debug(f"Calculated SNR: {snr_value:.2f}")

        return snr_value

    async def test_prompt_tokens(self, audio_file: str, prompt: str):
        # Example prompt (you can replace this with your own)
        prompt = "Baseball game commentary transcription. Focus on accurate recognition of key terms like pitch, strikeout, home run, etc."

        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Set up an example of an audio file for transcription (replace 'audio_file_path' with your actual file)
        audio_file = audio_file

        # Transcribe the audio with the initial_prompt
        segments_with_prompt, _ = self.model.transcribe(audio_file, initial_prompt=prompt)

        # Count the tokens generated by the transcription (with prompt)
        transcription_tokens_with_prompt = 0
        for segment in segments_with_prompt:
            transcription_tokens_with_prompt += len(self.tokenizer.encode(segment.text))

        # Total tokens (prompt + transcription with prompt)
        total_tokens_with_prompt = len(prompt_tokens) + transcription_tokens_with_prompt

        logger.info(f"[{self.worker_name}] --- Results WITH Initial Prompt ---")
        logger.info(f"[{self.worker_name}] Number of tokens in the prompt: {transcription_tokens_with_prompt}")
        logger.info(f"[{self.worker_name}] Number of tokens in the prompted transcription: {transcription_tokens_with_prompt}")
        logger.info(f"[{self.worker_name}] Total tokens (prompt + prompted transcription): {total_tokens_with_prompt}")

        # Transcribe the audio without the initial_prompt
        segments_without_prompt, _ = self.model.transcribe(audio_file)

        # Count the tokens generated by the transcription (without prompt)
        transcription_tokens_without_prompt = 0
        for segment in segments_without_prompt:
            transcription_tokens_without_prompt += len(self.tokenizer.encode(segment.text))

        # Total tokens (transcription without prompt)
        total_tokens_without_prompt = transcription_tokens_without_prompt

        logger.info(f"[{self.worker_name}] --- Results WITHOUT Initial Prompt ---")
        logger.info(f"[{self.worker_name}] Number of tokens in the unprompted transcription: {transcription_tokens_without_prompt}")
        logger.info(f"[{self.worker_name}] Total tokens (transcription only): {total_tokens_without_prompt}")

        # Check if total with prompt exceeds the 255-token limit
        if total_tokens_with_prompt > 255:
            logger.info(f"[{self.worker_name}] Warning: Total tokens (with prompt) exceed the 255-token limit by {total_tokens_with_prompt - 255} tokens.")
        else:
            logger.info(f"[{self.worker_name}] The total token count (with prompt) is within the 255-token limit.")

        # Check if total without prompt exceeds the 255-token limit
        if total_tokens_without_prompt > 255:
            logger.info(f"[{self.worker_name}] Warning: Total tokens (without prompt) exceed the 255-token limit by {total_tokens_without_prompt - 255} tokens.")
        else:
            logger.info(f"[{self.worker_name}] The total token count (without prompt) is within the 255-token limit.")

        # Check if the total tokens with prompt are less than the total tokens without prompt
        if total_tokens_with_prompt < total_tokens_without_prompt:
            logger.info(f"[{self.worker_name}] Warning: The total token count (with prompt) is less than the total token count (without prompt) by {total_tokens_without_prompt - total_tokens_with_prompt} tokens.")
        else:
            logger.info(f"[{self.worker_name}] The total token count (with prompt) is greater than or equal to the total token count (without prompt).")
