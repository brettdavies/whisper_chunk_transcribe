# `Utils.py`

## Design Considerations and Enhancements

- **Modular Utility Functions**: The `Utils` class provides a set of static utility functions for validating directories, reading CSV files, interacting with command-line arguments, and extracting information from file paths. These functions are designed to be reusable and flexible, enhancing modularity across the codebase.
- **Error Handling and Logging**: The module integrates comprehensive logging using `loguru` to track file operations, directory validation, and data extraction. This makes it easier to identify and troubleshoot potential issues during execution.
- **Database Integration**: Several utility functions interact with the database, allowing for seamless insertion of data (e.g., prompt terms) into the appropriate database tables.

## Module Overview

The `Utils.py` module contains utility functions for common tasks such as directory validation, file reading, and database operations. The functions are designed to be asynchronous and integrate with the existing logging and database infrastructure. These utilities help with handling files, validating paths, reading data from CSV files, and extracting metadata from file paths.

### Key Libraries and Dependencies

- **`pandas`**: Utilized for reading CSV files into DataFrames.
  - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

---

## Methods

### `validate_and_create_dir`

This function validates the provided directory path. If the path is not provided or is invalid, the function attempts to fetch the directory path from an environment variable. Optionally, it can create the directory if it does not exist.

#### Method: `validate_and_create_dir(dir_path: Path, env_var_name: str, create: bool) -> Path`

Validates or creates the directory specified by `dir_path`. If the path is not provided or invalid, the function attempts to use the value of an environment variable (`env_var_name`). If the `create` flag is set to `True`, it creates the directory if it does not exist.

```python
@staticmethod
async def validate_and_create_dir(dir_path: Path, env_var_name: str, create: bool) -> Path:
    """
    Validate directory, attempt to fetch from environment variable if not provided. Create if bool is True.
    
    Args:
        dir_path (Path): The directory path to validate or create.
        env_var_name (str): The name of the environment variable to check for the directory path if the input path is invalid.
        create (bool): Whether to create the directory if it doesn't exist.

    Returns:
        Path: The validated directory path or `False` if validation fails or the directory cannot be created.
    """
```

### `read_prompt_terms_from_file`

This function reads prompt terms from a CSV file into a Pandas DataFrame. The terms are used for transcription prompt generation.

#### Method: `read_prompt_terms_from_file(prompt_terms_file: str, worker_name: str) -> pd.DataFrame`

```python
@staticmethod
async def read_prompt_terms_from_file(prompt_terms_file: str, worker_name: str) -> pd.DataFrame:
    """
    Reads prompt terms from a CSV file into a Pandas DataFrame.

    Parameters:
        prompt_terms_file (str): The path of the CSV containing prompt terms.
        worker_name (str): Identifier for the worker.

    Returns:
        pd.DataFrame: DataFrame containing the prompt terms.
    """
```

### `read_prompt_terms_from_cli_argument_insert_db`

This function reads a file path for prompt terms from command-line arguments, loads the terms into a Pandas DataFrame, and inserts the data into the database.

#### Method: `read_prompt_terms_from_cli_argument_insert_db(prompt_terms_file: str) -> None`

```python
@staticmethod
async def read_prompt_terms_from_cli_argument_insert_db(prompt_terms_file: str) -> None:
    """
    This function reads a file path for prompt terms from command-line arguments, loads the terms into a Pandas DataFrame, and inserts the data into the database.

    Args:
        prompt_terms_file (str): The file path containing prompt terms (usually a CSV).

    Returns:
        None
    """
```

### `extract_video_info_filepath`

This function extracts video and format IDs from a file path string. The IDs are expected to be embedded within curly braces in the file path.

#### Method: `extract_video_info_filepath(filepath: str) -> Tuple[str, str]`

```python
@staticmethod
async def extract_video_info_filepath(filepath: str) -> Tuple[str, str]:
    """
    Extracts video and format IDs from a file path string. The IDs are formatted within curly braces, for example: `yt-{video_id}` or `fid-{format_id}`.

    Args:
        filepath (str): The file path containing video and format information.

    Returns:
        Tuple[str, str]: A tuple containing video_id and format_id.
    """
```

---

## Usage Example

Here’s an example of how to use the `Utils` class for common tasks such as validating directories, reading files, and extracting information from a file path.

```python
from pathlib import Path
from utils import Utils

# Example 1: Validate and create directory
dir_path = Path("/path/to/directory")
validated_dir = await Utils.validate_and_create_dir(dir_path, "DATA_DIR", create=True)

# Example 2: Read prompt terms from a CSV file
prompt_terms_file = "path/to/prompt_terms.csv"
prompt_terms_df = await Utils.read_prompt_terms_from_file(prompt_terms_file, worker_name="worker_1")

# Example 3: Extract video info from file path
file_path = "/data/{yt-1234}/fid-5678}"
video_id, format_id = await Utils.extract_video_info_filepath(file_path)
print(f"Video ID: {video_id}, Format ID: {format_id}")
```
