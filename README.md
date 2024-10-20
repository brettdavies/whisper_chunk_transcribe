# Whisper Chunk Transcribe

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

## Prerequisites

Before using `yt_dlp_async`, ensure that you have the following prerequisites installed:

- **Python 3.12 or higher**
- **Poetry**
- **yt-dlp**
- **Access to a PostgreSQL database**

## Documentation

Detailed documentation for each module is available in the `docs` directory:

- [audio_processor.py](docs/audio_processor.md): Provides functions for processing audio files, including loading, transforming, and saving audio data for further analysis or transcription.

- [database.py](docs/database.md): Manages database operations, including establishing SSH tunnels, handling database connections, and executing queries using a singleton pattern for efficient connection management.

- [db_populate_data.sql](docs/db_populate_data.md): SQL script for populating the database with initial data, including sample records, default values, and test data for development purposes.

- [db_schema.sql](docs/db_schema.md): Defines the database schema, including tables, relationships, indexes, constraints, and triggers to enforce data integrity and support application functionality.

- [experiment_runner.py](docs/experiment_runner.md): Orchestrates the execution of experiments, managing test cases, collecting results, and storing data for analysis.

- [helper_classes.py](docs/helper_classes.md): Provides helper classes and data structures used throughout the application to facilitate data handling and processing.

- [helper_scripts.py](docs/helper_scripts.md): Contains utility scripts for tasks such as data migration, cleanup, batch processing, and other auxiliary operations.

- [prepare_audio.py](docs/prepare_audio.md): Prepares audio files for processing, including splitting, normalization, noise reduction, and feature extraction.

- [process_audio_wrapper.py](docs/process_audio_wrapper.md): Provides a high-level interface for processing audio data, wrapping lower-level functions to handle audio processing tasks efficiently.

- [transcription_processor.py](docs/transcription_processor.md): Manages transcription of audio data, including sending audio to transcription services, processing responses, and handling errors.

- [transcription_wrapper.py](docs/transcription_wrapper.md): Provides a wrapper for the transcription process, managing the flow of audio data through transcription and post-processing steps.

- [utils.py](docs/utils.md): Offers utility functions and helpers used across the application, such as logging setup, configuration parsing, and common data transformations.

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
