# Standard Libraries
import os
from pathlib import Path
from typing import Tuple

# Logging
from loguru import logger

# Third Party Libraries
import pandas as pd

# First Party Libraries
from .database import DatabaseOperations

class Utils:
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
        if not dir_path or not dir_path.is_dir():
            dir_path = Path(os.getenv(env_var_name))
            if not dir_path:
                logger.error(f"{env_var_name} is not set")
                return False
            elif create:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create directory at {dir_path}: {e.strerror}")
                    return False
        return dir_path

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
        try:
            df = pd.read_csv(prompt_terms_file)
            logger.info(f"[{worker_name}] CSV file '{prompt_terms_file}' read into DataFrame successfully.")
            return df
        except Exception as e:
            logger.error(f"[{worker_name}] Failed to read CSV file '{prompt_terms_file}': {e}")
            raise e

    @staticmethod
    async def read_prompt_terms_from_cli_argument_insert_db(prompt_terms_file: str) -> None:
        """
        This function reads a file path for prompt terms from command-line arguments, loads the terms into a Pandas DataFrame, and inserts the data into the database.

        Args:
            prompt_terms_file (str): The file path containing prompt terms (usually a CSV).

        Returns:
            None
        """
        try:
            logger.info(f"Reading prompt terms from file: {prompt_terms_file}")
            prompt_terms_file = prompt_terms_file.strip()
            file_path = Path(prompt_terms_file)

            # Check if the file exists as is
            if not file_path.is_file():
                from dotenv import load_dotenv
                load_dotenv()
                data_base_dir = Path(os.getenv("DATA_BASE_DIR"))
                file_path = data_base_dir / file_path
                logger.info(f"File not found. Trying with execution path: {file_path}")
                if not file_path.is_file():
                    logger.error(f"Prompt terms file does not exist: {prompt_terms_file} or {file_path}")
                    raise FileNotFoundError(f"Prompt terms file not found: {prompt_terms_file}")

            # Get the file extension
            file_extension = file_path.suffix
            logger.info(f"File extension: {file_extension}")
            if file_extension.lower() == '.csv':
                # Read the CSV into a pandas DataFrame
                prompt_terms_df = await Utils.read_prompt_terms_from_file(str(file_path))
                
                # Insert the DataFrame into the database
                await DatabaseOperations.insert_prompt_terms_bulk(prompt_terms_df)
            else:
                logger.error(f"{file_extension.lstrip('.').upper()} is not an accepted format. Please use CSV.")
        except Exception as e:
            logger.error(f"Error reading prompt_terms from CLI argument: {e}")
            raise e    

    @staticmethod
    async def extract_video_info_filepath(filepath: str) -> Tuple[str, str]:
        file_name_parts = filepath.split('{')
        video_id = None
        a_format_id = None
        for part in file_name_parts:
            if part.startswith('yt-'):
                video_id = part.split('yt-')[1].split('}')[0]
            if part.startswith('fid-'):
                a_format_id = part.split('fid-')[1].split('}')[0]
            if video_id and a_format_id:
                break
        return video_id, a_format_id
