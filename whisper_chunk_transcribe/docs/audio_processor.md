# `audio_processor.py`

## Design Considerations and Enhancements

- **Extensibility**: The `AudioProcessor` class is modular and flexible, allowing for easy integration of additional audio processing techniques such as format transformation, noise reduction, and voice activity detection (VAD).
- **Asynchronous Processing**: `asyncio` is used to manage I/O-bound operations asynchronously, optimizing performance for large audio datasets.
- **Noise Reduction and Voice Activity Detection (VAD)**: Incorporates advanced methods for reducing background noise and detecting speech activity in audio, preparing audio files for further analysis or transcription.
- **External Libraries**: Libraries such as `pydub`, `noisereduce`, `pyannote.audio`, and `scipy` provide essential functionality for audio manipulation and analysis.
- **Database Integration**: The class interacts with `DatabaseOperations` to log audio processing steps and save results into the database for traceability.

## Module Overview

The `audio_processor.py` module provides various methods to process audio files. It handles audio format transformation, noise reduction, voice activity detection (VAD), and related operations. The primary tasks include loading audio, applying filters, segmenting based on VAD, and calculating Signal-to-Noise Ratio (SNR) for audio segments.

### Key Libraries and Dependencies

- **`pydub`**: A library for manipulating audio, used for format conversion and handling audio segments.
  - [Pydub Documentation](https://pydub.com/)
- **`torch`**: Deep learning library used in conjunction with VAD models.
  - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- **`pyannote.audio`**: Provides pre-trained pipelines for tasks such as voice activity detection (VAD).
  - [Pyannote Audio Documentation](https://pyannote.github.io/pyannote-audio/)
- **`noisereduce`**: Reduces noise in audio files using non-stationary noise reduction.
  - [NoiseReduce Documentation](https://pypi.org/project/noisereduce/)
- **`scipy.signal.wiener`**: Wiener filter for noise reduction.
  - [Wiener Filter Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html)

---

## Classes and Methods

### `AudioProcessor`

The `AudioProcessor` class is responsible for various audio processing tasks, including audio format conversion, noise reduction, voice activity detection (VAD), segmentation, and signal-to-noise ratio (SNR) calculation.

#### Constructor: `__init__(self, model_for_vad: Path, source_file_path: Path, output_dir: Path, worker_name: str, db_ops: DatabaseOperations) -> None`

Initializes an instance of the `AudioProcessor` class.

---

### Key Methods

#### Method: `transform_source2wav(self) -> None`

Transforms the source audio file to WAV format, applies noise reduction, and saves the processed audio.

```python
async def transform_source2wav(self) -> None:
    """
    Complete audio processing pipeline:
    1. Resamples the audio to 16 kHz.
    2. Applies a Wiener filter to reduce noise.
    3. Uses non-stationary noise reduction to further clean the audio.
    4. Saves both the original and processed audio as WAV files.

    Returns:
        None
    """
```

---

#### Method: `load_vad_pipeline(self) -> VoiceActivityDetection`

Loads the Voice Activity Detection (VAD) pipeline using the specified model.

- **How it Works**:
  - Loads a pre-trained VAD model using `pyannote.audio`.
  - The model can be run on a GPU or CPU depending on availability.

- **Returns**: An instance of `VoiceActivityDetection`.

```python
async def load_vad_pipeline(self) -> VoiceActivityDetection:
    """
    Load the Voice Activity Detection (VAD) pipeline using the specified model.

    Returns:
        VoiceActivityDetection: The loaded VAD pipeline.
    """
```

---

#### Method: `extract_segments_from_audio_file(self) -> None`

Extracts segments from the audio file using VAD, storing the results in a text file.

- **How it Works**:
  - Loads the processed audio file.
  - Applies the VAD pipeline to segment the audio based on speech detection.
  - Saves the VAD segments to a text file for later use.

- **Returns**: None

```python
async def extract_segments_from_audio_file(self) -> None:
    """
    Extracts segments from the audio file using VAD, storing the results in a text file.

    Returns:
        None
    """
```

---

#### Method: `segment_audio_based_on_vad(self) -> None`

Segments the audio file based on VAD results, saving each segment as a separate file.

- **How it Works**:
  - Iterates over the segments detected by VAD.
  - Saves each segment as an individual WAV file in the `segments` directory.
  - Updates the segment DataFrame for database logging.

```python
async def segment_audio_based_on_vad(self) -> None:
    """
    Segments the audio file based on VAD (Voice Activity Detection) results.
    Each VAD segment is saved as a separate audio file.

    Returns:
        None
    """
```

---

#### Method: `load_and_resample_audio(self, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]`

Loads and resamples the source audio file to the target sample rate.

- **How it Works**:
  - Loads the source audio file using `pydub`.
  - Resamples the audio to the target sample rate (default 16 kHz).
  - Returns the resampled waveform as a numpy array and the sample rate.

- **Returns**: A tuple containing the resampled waveform and sample rate.

```python
async def load_and_resample_audio(self, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load the source audio file and resample it to the target sample rate.

    Args:
        target_sample_rate (int): The desired sample rate. Default is 16000 (16kHz).

    Returns:
        tuple: (waveform, sample_rate)
    """
```

---

#### Method: `apply_wiener_filter(self, waveform: np.ndarray, epsilon: float = 1e-10) -> np.ndarray`

Applies a Wiener filter to the waveform for noise reduction.

- **How it Works**:
  - Uses `scipy.signal.wiener` to reduce noise in the waveform.
  - Adds a small epsilon value to avoid division by zero.

- **Returns**: The cleaned waveform as a numpy array.

```python
async def apply_wiener_filter(self, waveform: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Apply a Wiener filter to the input waveform to reduce noise.

    Args:
        waveform (np.ndarray): The input waveform.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: The cleaned waveform.
    """
```

---

#### Method: `apply_non_stationary_noise_reduction(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray`

Applies non-stationary noise reduction to the waveform using the `noisereduce` library.

- **How it Works**:
  - Reduces background noise in the audio using the `noisereduce` algorithm.
  - The input waveform and sample rate are required.

- **Returns**: The cleaned waveform as a numpy array.

```python
async def apply_non_stationary_noise_reduction(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply non-stationary noise reduction to the waveform.

    Args:
        waveform (np.ndarray): The input waveform.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        np.ndarray: The cleaned waveform.
    """
```

---

#### Method: `save_waveform_to_wav(self, waveform: np.ndarray, sample_rate: int, is_original: bool, amplification_factor: float = 2.0) -> None`

Saves the waveform as a WAV file with the specified sample rate and amplification.

- **How it Works**:
  - Saves the waveform using `scipy.io.wavfile` after amplifying the signal.
  - Supports saving both the original and processed versions of the audio.

- **Returns**: None

```python
async def save_waveform_to_wav(self, waveform: np.ndarray, sample_rate: int, is_original: bool, amplification_factor: float = 2.0) -> None:
    """
    Save the waveform as a WAV file with amplification.

    Args:
        waveform (np.ndarray): The waveform data.
        sample_rate (int): The sample rate of the waveform.
        is_original (bool): Flag indicating if it's the original audio.
        amplification_factor (float): Factor to amplify the waveform.

    Returns:
        None
    """
```

---

#### Method: `calculate_snr(self) -> None`

Calculates the Signal-to-Noise Ratio (SNR) for each audio segment and updates the database.

- **How it Works**:
  - Iterates through audio segments in batches.
  - Calculates the SNR for each segment using raw and processed audio files.
  - Updates the SNR values in the database.

- **Returns**: None

```python
async def calculate_snr(self) -> None:
    """
    Calculate the Signal-to-Noise Ratio (SNR) for each segment and update the database.

    Returns:
        None
    """
```

---

#### Method: `compute_snr(self, raw_audio_path: Path, processed_audio_path: Path) -> float`

Computes the Signal-to-Noise Ratio (SNR) between the raw and processed audio files.

- **How it Works**:
  - Loads the raw and processed audio files.
  - Computes the SNR using the formula: \( SNR = 10 \log_{10}( \frac{signal\_power}{noise\_power}) \).

- **Returns**: The calculated SNR value as a float.

```python
async def compute_snr(self, raw_audio_path: Path, processed_audio_path: Path) -> float:
    """
    Compute the Signal-to-Noise Ratio (SNR) between raw and processed audio files.

    Args:
        raw_audio_path (Path): The path to the raw audio file.
        processed_audio_path (Path): The path to the processed audio file.

    Returns:
        float: The calculated SNR.
    """
```