# README for the Whisper Chunk Transcribe code

## Introduction

This poetry application provides an advanced framework for processing audio data, running experiments on audio transcriptions, and managing related resources such as databases and utility functions. The framework is designed with extensibility, flexibility, and performance in mind, allowing users to seamlessly integrate audio transcription and experiment workflows.

The system is composed of several key components:

- **Audio Processing**: Modules for preparing and processing audio data, ensuring it is ready for transcription.
- **Transcription Workflow**: Classes and methods to manage the transcription of audio files using various engines and handling the necessary pre- and post-processing steps.
- **Experimentation Framework**: An organized structure for running experiments, handling experiment configurations, and automating processes.
- **Helper and Utility Functions**: A suite of helper functions and classes that enhance the flexibility of the core modules, improving overall code efficiency.
- **Database Management**: Tools for managing the storage and retrieval of audio, transcription data, and experiment results in a structured and efficient manner.

## Key Features

- **Modular Design**: The project is divided into distinct, self-contained modules that handle different aspects of audio processing and transcription, making the system easy to extend and modify.
- **Scalability**: The framework is designed to handle a wide range of audio processing workloads, from small datasets to large-scale experiments involving multiple transcription engines.
- **Configurable Experimentation**: Users can easily configure and run experiments with various settings, allowing for in-depth testing and analysis of transcription methods.
- **Robust Error Handling**: The system includes extensive error handling mechanisms, ensuring the smooth execution of processes and providing detailed error reports for debugging.
- **Optimized Audio Processing**: The audio processing components are optimized for performance, including enhancements to reduce processing time without sacrificing accuracy.

## Architecture

The system follows a modular architecture where each module serves a specific role. The following are the primary modules:

1. **Audio Processing [`audio_processor.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/audio_processor.py)([Docs](/whisper_chunk_transcribe/docs/audio_processor.md))**:

   - Prepares audio data for transcription by normalizing, converting, and validating files.
   
2. **Transcription Workflow [`transcription_processor.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/transcription_processor.py)([Docs](/whisper_chunk_transcribe/docs/transcription_processor.md)), [`transcription_wrapper.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/transcription_wrapper.py)([Docs](/whisper_chunk_transcribe/docs/transcription_wrapper.md))**:
   - Manages the transcription of audio data, integrating with different transcription engines and handling pre- and post-processing tasks.

3. **Experimentation [`experiment_runner.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/experiment_runner.py)([Docs](/whisper_chunk_transcribe/docs/experiment_runner.md))**:
   - Provides tools to set up and execute experiments involving audio data and transcription engines.

4. **Helper and Utility Modules [`helper_classes.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/helper_classes.py)([Class Docs](/whisper_chunk_transcribe/docs/helper_classes.md), [Script Docs](/whisper_chunk_transcribe/docs/helper_scripts.md)), [`utils.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/utils.py)([Docs](/whisper_chunk_transcribe/docs/utils.md))**:
   - Offers general-purpose classes and utility functions to assist with various tasks across the system.

5. **Database Management [`database.py`](/whisper_chunk_transcribe/whisper_chunk_transcribe/database.py)([Docs](/whisper_chunk_transcribe/docs/database.md))**:
   - Handles database operations such as saving and retrieving transcription data and experiment results.

## Workflow

The typical workflow of the project involves the following steps:

1. **Audio Preparation**: Audio files are loaded, processed, and prepared for transcription.
2. **Transcription Execution**: The audio files are passed through the transcription system, which may involve one or more transcription engines.
3. **Experimentation**: Experiments are set up to test the performance of various transcription approaches. Results are logged and saved for further analysis.
4. **Result Management**: Transcription results and experiment logs are stored in a database for easy retrieval and comparison.

## Environment Variables

The project relies on a set of environment variables to configure various aspects of its behavior, including logging, file paths, model locations, and database connection settings. The following are key environment variables that need to be configured:

### Logging Configuration
- `LOG_LEVEL`: Specifies the logging level (e.g., INFO, DEBUG) for controlling the verbosity of the system's logs.

### Application-Specific Configuration
- `OUTPUT_DIR`: The directory where output files, such as processed audio or transcription results, will be stored.
- `DATA_BASE_DIR`: The base directory for data files used throughout the application.
- `AUDIO_FILE_DIR`: Directory where audio files to be processed are located.
- `CHUNK_SIZE`: The size (in seconds) of the audio chunks for processing.
- `MODEL_BASE_DIR`: Base directory where model files are located.
- `MODEL_FOR_VAD`: Path to the Voice Activity Detection (VAD) model.
- `MODEL_FOR_TRANSCRIBE`: Path to the model used for transcription tasks.

### SSH Tunnel Configuration
- `SSH_HOST`: The SSH host used for tunneling connections (if necessary).
- `SSH_PORT`: The port used for the SSH tunnel.
- `SSH_USER`: The SSH username for establishing the connection.
- `SSH_KEY_PATH`: The path to the SSH private key file for authentication.

### Database Configuration
- `DB_HOST`: The host for the database connection.
- `DB_USER`: The username for the database.
- `DB_PASSWORD`: The password for the database.
- `DB_NAME`: The name of the database to connect to.

Ensure these environment variables are properly set in your system before running the project to avoid misconfigurations and ensure smooth execution of processes.

## Attention to Detail

The project has been designed with a strong emphasis on code quality and maintainability:
- **Docstrings and Comments**: All classes and methods are well-documented, providing clear guidance on their purpose and usage.
- **Error Handling**: Extensive checks and balances ensure that the system can handle a variety of edge cases, providing users with meaningful feedback in case of errors.
- **Code Efficiency**: The codebase includes numerous optimizations aimed at reducing overhead and improving performance.
