# Standard Libraries
import os
import json
from dataclasses import dataclass

# CLI, Logging, Configuration
import fire
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
import numpy as np
import torch
from scipy.signal import wiener
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment
from pyannote.core import Annotation
from pyannote.audio.pipelines import VoiceActivityDetection
from faster_whisper import WhisperModel

# First Party Libraries
from .logger_config import LoggerConfig

# Configure loguru
LOGGER_NAME = "main"
LoggerConfig.setup_logger(LOGGER_NAME)

# Global device variable
device: str = "cpu" # or "cuda"

def load_and_resample_audio(m4a_file_path, target_sample_rate=16000) -> tuple[np.ndarray, int]:
    """
    Load the M4A audio file and resample it to the target sample rate.
    
    Parameters:
    - m4a_file_path (str): The file path of the M4A audio file.
    - target_sample_rate (int): The desired sample rate for the resampled audio. Default is 16000.
    
    Returns:
    - waveform (np.ndarray): The resampled audio waveform as a numpy array.
    - target_sample_rate (int): The actual sample rate of the resampled audio.
    """

    try:
        audio = AudioSegment.from_file(m4a_file_path, format="m4a")
        audio = audio.set_frame_rate(target_sample_rate)
        audio = audio.set_channels(1)  # Ensure the audio is mono
        waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Optional: Normalize the waveform to the range [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Debug: Log information about the waveform
        logger.debug(f"Waveform dtype: {waveform.dtype}")
        logger.debug(f"Waveform shape: {waveform.shape}")
        logger.debug(f"Waveform max value: {np.max(waveform)}")
        logger.debug(f"Waveform min value: {np.min(waveform)}")
        
        return waveform, target_sample_rate

    except Exception as e:
        logger.error(f"Error loading and resampling audio: {e}")
        return None, target_sample_rate

def apply_wiener_filter(waveform: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
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
    if not isinstance(waveform, np.ndarray):
        raise ValueError("waveform must be a numpy.ndarray")
    if not isinstance(epsilon, float):
        raise ValueError("epsilon must be a float")

    try:
        cleaned_waveform = wiener(waveform) + epsilon  # Avoid division by zero
        if np.all(cleaned_waveform == 0):
            logger.warning("Wiener filter output is all zeros.")
        return cleaned_waveform
    except Exception as e:
        logger.error(f"Error applying Wiener filter: {e}")
        return waveform

def apply_non_stationary_noise_reduction(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
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
    if not isinstance(waveform, np.ndarray):
        raise ValueError("waveform must be a numpy.ndarray")
    if not isinstance(sample_rate, int):
        raise ValueError("sample_rate must be an int")

    try:
        cleaned_waveform = nr.reduce_noise(y=waveform, sr=sample_rate)
        return cleaned_waveform
    except Exception as e:
        logger.error(f"Error applying non-stationary noise reduction: {e}")
        return waveform

def save_waveform_to_wav(waveform: np.ndarray, sample_rate: int, output_path: str, amplification_factor: float = 2.0) -> None:
    """
    Save the waveform as a WAV file with the specified sample rate using scipy.io.wavfile.
    
    Parameters:
    - waveform: np.ndarray
        The waveform data to be saved as a WAV file.
    - sample_rate: int
        The sample rate of the waveform.
    - output_path: str
        The path where the WAV file will be saved.
    - amplification_factor: float, optional
        The amplification factor to apply to the waveform. Default is 2.0.
    """
    if not isinstance(waveform, np.ndarray):
        raise ValueError("waveform must be a numpy.ndarray")

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
        wavfile.write(output_path, sample_rate, waveform_int16)
        logger.info(f"Saved WAV file to {output_path}")
    except Exception as e:
        logger.error(f"Error saving WAV file: {e}")

def transform_mp4_wav(m4a_file_path, wav_file_path) -> None:
    """
    Complete audio processing pipeline:
    1. Load and resample audio to 16 kHz.
    2. Apply Wiener filtering.
    3. Apply Non-stationary Noise Reduction.
    4. Save the processed audio as a 16 kHz WAV file.
    """
    try:
        # Load and resample the audio
        waveform, sample_rate = load_and_resample_audio(m4a_file_path)
        
        # Apply Wiener filter
        cleaned_waveform = apply_wiener_filter(waveform)
        
        # Apply Non-stationary Noise Reduction
        reduced_noise_waveform = apply_non_stationary_noise_reduction(cleaned_waveform, sample_rate)
        
        # Save the processed waveform to a WAV file
        save_waveform_to_wav(reduced_noise_waveform, sample_rate, wav_file_path)
    except Exception as e:
        logger.error(f"Error in transform_mp4_wav: {e}")

def process_audio_files_into_segments(wav_file_path: str) -> Annotation:
    """
    Process the given WAV file and segment it using voice activity detection (VAD).

    Parameters:
    - wav_file_path: str
        The path to the WAV file to be processed.

    Returns:
    - Annotation
        An Annotation object containing the segments of detected speech.
    """
    try:
        # Load the .wav file
        audio = AudioSegment.from_file(wav_file_path, format="wav")

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
        waveform_tensor = torch.tensor(waveform).to(device)

        # Load the VAD model with the specified device (GPU)
        vad_model_path = "/media/bigdaddy/data/cache_model/pyannote_segmentation-3.0/pytorch_model.bin"
        pipeline = VoiceActivityDetection(segmentation=vad_model_path, device=torch.device(device))

        # Define hyperparameters for VAD
        HYPER_PARAMETERS = {
            "min_duration_on": 0.2,   # Remove speech regions shorter than this many seconds
            "min_duration_off": 0.2,  # Fill non-speech regions shorter than this many seconds
            # "onset": 0.5,             # Start of speech detection threshold
            # "offset": 0.5,            # End of speech detection threshold
            # "threshold": 0.6,         # Threshold for speech detection in noisy environments
        }
        pipeline.instantiate(HYPER_PARAMETERS)

        # Perform VAD on the waveform tensor
        vad: Annotation = pipeline({"waveform": waveform_tensor, "sample_rate": 16000})
        logger.info(f"VAD results:\n{vad}")

        # Generate the output file path for the VAD results
        output_txt_path = os.path.splitext(wav_file_path)[0] + ".txt"

        # Write the VAD results to a text file
        with open(output_txt_path, 'w') as f:
            f.write(str(vad))

        logger.info(f"Processed VAD for {wav_file_path}, found {len(vad)} segments.")
        return vad
    
    except Exception as e:
        logger.error(f"Error processing audio file into segments: {e}")
        return Annotation()

def chunk_audio_based_on_vad(vad_result: Annotation, wav_file_path: str) -> str:
    """
    Chunks the audio file based on VAD (Voice Activity Detection) results.
    Each VAD segment is saved as a separate audio file in a subfolder called 'chunks'.

    Parameters:
    - vad_result: Annotation
        An Annotation object containing VAD results.
    - wav_file_path: str
        Path to the original audio file.

    Returns:
    - chunk_file_path: str
        The filepath where the chunks were saved.
    """
    if not isinstance(vad_result, Annotation):
        raise ValueError("vad_result must be an Annotation object")
    try:
        # Create a directory for the chunks
        chunk_dir = os.path.join(os.path.dirname(wav_file_path), "chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        # Load the original audio file
        audio = AudioSegment.from_file(wav_file_path)

        # Iterate over the segments in the VAD results and create audio chunks
        for idx, segment in enumerate(vad_result.itersegments()):
            start_time = segment.start
            end_time = segment.end

            # Convert start_time and end_time from seconds to milliseconds
            start_ms = start_time * 1000
            end_ms = end_time * 1000

            # Extract the audio chunk
            chunk = audio[start_ms:end_ms]

            # Generate the chunk file name
            chunk_file_name = f"{idx + 1:05}.wav"
            chunk_file_path = os.path.join(chunk_dir, chunk_file_name)

            # Export the chunk as a WAV file
            chunk.export(chunk_file_path, format="wav")

            logger.info(f"Chunk {chunk_file_name} saved at {chunk_file_path}")

        return chunk_dir

    except Exception as e:
        logger.info(f"Error during chunking: {e}")
        return None

def transcribe_audio(directory: str) -> None:
    """
    Transcribe all WAV files in the given directory using the faster-whisper model.
    The function writes the transcription, word-level confidence scores, 
    and segment-level confidence scores to separate files.

    Parameters:
    - directory: str
        The path to the directory containing WAV files.

    Returns:
    None
    """
    try:
        if not os.path.isdir(directory):
            logger.error(f"Directory does not exist: {directory}")
            return

        # Load the Whisper model from the local directory
        model_dir = "/media/bigdaddy/data/cache_model/faster-distil-whisper-medium.en"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(model_dir, device=device)

        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                file_path = os.path.join(directory, filename)
                logger.info(f"Processing file: {file_path}")

                try:
                    # Transcribe the audio file
                    segments_generator, info = model.transcribe(file_path, beam_size=5, word_timestamps=True)

                    # Convert generator to list
                    segments = list(segments_generator)
                    logger.debug(f"Segments:\n{segments}")

                    if not segments:
                        logger.error(f"No segments found for file: {file_path}")
                        continue

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
                            logger.warning(f"No word-level details for segment: {segment.text}")

                        segment_confidences.append({
                            "start": segment.start,
                            "end": segment.end,
                            "complement_avg_logprob": (1 + segment.avg_logprob)
                        })

                    # Write the transcription text to a file
                    transcription_file = file_path.replace(".wav", ".txt")
                    with open(transcription_file, 'w') as f:
                        f.write("\n".join(transcription_text))
                    logger.info(f"Transcription written to: {transcription_file}")

                    # Write word-level confidence scores to a file
                    word_confidence_file = file_path.replace(".wav", "_word_confidences.json")
                    with open(word_confidence_file, 'w') as f:
                        json.dump([wc._asdict() for wc in word_confidences], f, indent=4)
                    logger.info(f"Word-level confidence scores written to: {word_confidence_file}")

                    # Write segment-level confidence scores to a file
                    segment_confidence_file = file_path.replace(".wav", "_segment_confidences.json")
                    with open(segment_confidence_file, 'w') as f:
                        json.dump(segment_confidences, f, indent=4)
                    logger.info(f"Segment-level confidence scores written to: {segment_confidence_file}")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error during transcription: {e}")

@dataclass(slots=True)
class Fetcher:
    """
    A class that fetches audio files, processes them, and transcribes the audio.
    Methods:
    - main(directory=None): Fetches audio files from the specified directory or the default directory, processes the audio files, and transcribes the audio.
    """
    def main(self, directory: str = None) -> None:
        global device

        if not directory or not os.path.isdir(directory):
            try:
                load_dotenv()
                directory = os.getenv("AUDIO_DIRECTORY")
                if not directory or not os.path.isdir(directory):
                    raise ValueError("AUDIO_DIRECTORY is not set or is invalid")
            except Exception as e:
                logger.error(f"AUDIO_DIRECTORY is not set or error occurred: {e}")
                return
        
        logger.info(f"directory: {directory}")

        # Determine the appropriate device (GPU or CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Processor: {device}")

        m4a_file_path = "data/audio/lan at cin - 2010.04.22 - [en][H02M50S12][44100][No DRC][mp4a.40.2][3.0][medium]{fid-140}{e-300422117}{yt-05PEUtD_Sg0}.m4a"
        wav_file_path = "data/audio/lan at cin - 2010.04.22 - [en][H02M50S12][44100][No DRC][mp4a.40.2][3.0][medium]{fid-140}{e-300422117}{yt-05PEUtD_Sg0}.wav"
        
        try:
            # Call the transform function
            transform_mp4_wav(m4a_file_path, wav_file_path)

            # Call the segmenting function
            vad: Annotation = process_audio_files_into_segments(wav_file_path)

            # Call the chunking function
            chunk_file_path = chunk_audio_based_on_vad(vad, wav_file_path)
            if not chunk_file_path:
                raise ValueError("Chunk file path is invalid")

            # Transcribe the audio
            transcribe_audio(chunk_file_path)

        except Exception as e:
            logger.error(f"Error in processing: {e}")

def cmd():
    fire.Fire(Fetcher)

if __name__ == "__main__":
    fire.Fire(Fetcher)
