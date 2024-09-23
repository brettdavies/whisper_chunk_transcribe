# Standard Libraries
import os
import psycopg2
from psycopg2 import pool
from typing import Any, List, Optional
from contextlib import contextmanager
from psycopg2.extras import DictCursor
from sshtunnel import SSHTunnelForwarder

# Logging and Configuration
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
import pandas as pd
import threading

# Load environment variables from .env file
load_dotenv()

class SingletonMeta(type):
    """
    A thread-safe implementation of Singleton.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        # First, check if an instance already exists without acquiring the lock for performance
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class DatabaseOperations(metaclass=SingletonMeta):
    def __init__(self) -> None:
        """
        Initialize the SSH tunnel and the psycopg2 connection pool.
        This method is called only once due to the Singleton pattern.
        """
        # Prevent re-initialization in case __init__ is called multiple times
        if hasattr(self, '_initialized') and self._initialized:
            logger.debug("DatabaseOperations instance already initialized.")
            return

        # SSH and database configuration
        ssh_host = os.environ.get('SSH_HOST')
        ssh_port = int(os.environ.get('SSH_PORT', 22))
        ssh_user = os.environ.get('SSH_USER')
        ssh_key_path = os.environ.get('SSH_KEY_PATH')
        remote_bind_address = ('127.0.0.1', 5432)

        db_user = os.environ.get('DB_USER')
        db_password = os.environ.get('DB_PASSWORD')
        db_name = os.environ.get('DB_NAME')

        # Validate required SSH and DB configurations
        required_vars = ['SSH_HOST', 'SSH_PORT', 'SSH_USER', 'SSH_KEY_PATH', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables for SSH or DB: {', '.join(missing)}")
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

        try:
            # Initialize the SSH tunnel
            self.tunnel = SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key_path,
                remote_bind_address=remote_bind_address
            )
            self.tunnel.start()
            logger.warning(f"SSH tunnel established on local port {self.tunnel.local_bind_port}.")

            # Initialize the connection pool
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,  # Adjust the max connections based on your requirements
                user=db_user,
                password=db_password,
                host='127.0.0.1',
                port=self.tunnel.local_bind_port,
                database=db_name
            )
            logger.warning("Database connection pool created successfully.")

            self._initialized = True  # Mark as initialized

        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            self.close()  # Ensure resources are cleaned up
            raise

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and clean up resources.
        """
        self.close()

    def close(self):
        """
        Close the connection pool and the SSH tunnel.
        """
        try:
            if hasattr(self, 'connection_pool') and self.connection_pool:
                self.connection_pool.closeall()
                logger.warning("Database connection pool closed.")

            if hasattr(self, 'tunnel') and self.tunnel:
                self.tunnel.stop()
                logger.warning("SSH tunnel closed.")
        except Exception as e:
            logger.error(f"Error during closing resources: {e}")

    @contextmanager
    def get_db_connection(self, worker_namne: Optional[str]) -> Any:
        """
        Context manager for database connection.
        Ensures that the connection is properly returned to the pool.
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            if conn is None:
                logger.error(f"{"["+worker_namne+"] " if worker_namne else None}Failed to obtain database connection from the pool.")
                raise psycopg2.OperationalError(f"{"["+worker_namne+"] " if worker_namne else None}Unable to obtain database connection.")
            yield conn
        except psycopg2.Error as e:
            logger.error(f"{"["+worker_namne+"] " if worker_namne else None}Error obtaining database connection: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
                logger.debug(f"{"["+worker_namne+"] " if worker_namne else None}Database connection returned to the pool.")

    def insert_prompt_terms_bulk(self, prompt_terms_df: pd.DataFrame) -> None:
        """
        Insert or update prompt terms in bulk from a pandas DataFrame into the 'prompt_terms' table.

        Parameters:
        - prompt_terms_df (pd.DataFrame): DataFrame containing prompt terms.

        Returns:
            None
        """
        required_columns = [
            'term',
            'in_game_usage_score',
            'general_speech_score',
            'impact_transcription_score',
            'confusion_potential_score',
            'tokens'
        ]

        # Validate DataFrame columns
        missing_columns = [col for col in required_columns if col not in prompt_terms_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in the DataFrame: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Prepare data as list of tuples
        data_tuples = list(prompt_terms_df[required_columns].itertuples(index=False, name=None))

        # Define the insert query with upsert using psycopg2's parameter placeholders (%s)
        insert_query = """
            INSERT INTO prompt_terms (
                term, 
                in_game_usage_score, 
                general_speech_score, 
                impact_transcription_score, 
                confusion_potential_score, 
                tokens
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (term) DO UPDATE SET
                in_game_usage_score = EXCLUDED.in_game_usage_score,
                general_speech_score = EXCLUDED.general_speech_score,
                impact_transcription_score = EXCLUDED.impact_transcription_score,
                confusion_potential_score = EXCLUDED.confusion_potential_score,
                tokens = EXCLUDED.tokens;
        """
        with self.get_db_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"Connection acquired successfully for insert_prompt_terms_bulk.")

                    # Execute the bulk insert using cursor.executemany
                    cursor.executemany(insert_query, data_tuples)
                    logger.info(f"Inserted/Updated {len(data_tuples)} records into 'prompt_terms' table.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"Error in insert_prompt_terms_bulk: {ex}")
                conn.rollback()
                logger.debug(f"Transaction rolled back due to error.")
                raise

    def insert_audio_segment_df(self, audio_segments_df: pd.DataFrame, worker_name: str) -> None:
        """
        Insert or update audio segments in bulk from a pandas DataFrame into the 'audio_segments' table.

        Parameters:
        - audio_segments_df (pd.DataFrame): DataFrame containing audio segments.

        Returns:
            None
        """
        required_columns = [
            'video_id',
            'segment_number',
            'start_time',
            'end_time',
            'raw_audio_path',
            'processed_audio_path'
        ]

        # Validate DataFrame columns
        missing_columns = [col for col in required_columns if col not in audio_segments_df.columns]
        if missing_columns:
            logger.error(f"[{worker_name}] Missing required columns in the DataFrame: {missing_columns}")
            raise ValueError(f"[{worker_name}] Missing required columns in the DataFrame: {missing_columns}")
        else:
            logger.debug(f"[{worker_name}] All required columns are present in the DataFrame.")

        # Prepare data as list of tuples
        data_tuples = list(audio_segments_df[required_columns].itertuples(index=False, name=None))
        logger.debug(f"[{worker_name}] Prepared {len(data_tuples)} data tuples for insertion into 'audio_segments'.")

        if not data_tuples:
            logger.warning(f"[{worker_name}] No data to insert into 'audio_segments'. Exiting method.")
            return

        # Define the insert query with upsert using psycopg2's parameter placeholders (%s)
        insert_query = """
            INSERT INTO audio_segments (
                video_id,
                segment_number,
                start_time,
                end_time,
                raw_audio_path,
                processed_audio_path
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, segment_number) DO UPDATE SET
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                raw_audio_path = EXCLUDED.raw_audio_path,
                processed_audio_path = EXCLUDED.processed_audio_path;
        """
        with self.get_db_connection(worker_namne=worker_name) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_audio_segment_df.")

                    # Execute the bulk insert using cursor.executemany
                    cursor.executemany(insert_query, data_tuples)
                    logger.info(f"[{worker_name}] Inserted / Updated {len(data_tuples)} segments.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_audio_segment_df: {ex}")
                conn.rollback()
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

    def get_exp_video_ids(self, experiment_id: int, worker_name: str) -> List[str]:
        """
        Retrieve the video IDs for a given experiment ID from the 'exp_experiment_videos' and 'yt_video_file' tables.

        Parameters:
        - experiment_id (int): The experiment ID to filter the video IDs.

        Returns:
            List[str]: A list of video local paths for the given experiment ID.
        """
        select_query = """
            SELECT files.local_path
            FROM exp_experiment_videos e_vids
            JOIN yt_video_file files ON e_vids.video_id = files.video_id
            WHERE e_vids.experiment_id = %s;
        """

        with self.get_db_connection(worker_namne=worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching video local paths for experiment ID: {experiment_id}")

                    # Execute the query
                    cursor.execute(select_query, (experiment_id,))
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} records from the database.")

                    # Extract 'local_path' from each record
                    return [record['local_path'] for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in get_exp_video_ids: {ex}")
                raise

    def get_audio_segments_no_snr(self, worker_name: str, batch_size: int) -> List[dict]:
        """
        Retrieve the audio segments with no SNR value from the 'audio_segments' table.

        Parameters:
        - worker_name (str): The name of the worker for logging purposes.
        - offset (int): The offset for pagination.
        - batch_size (int): The number of records to retrieve.

        Returns:
            List[dict]: A list of dictionaries containing segment IDs and file paths for segments with no SNR value.
        """
        select_query = """
            SELECT seg.segment_id, seg.raw_audio_path, seg.processed_audio_path
            FROM audio_segments AS seg
            WHERE seg.snr IS NULL
            ORDER BY seg.segment_id
            LIMIT %s;  -- Using LIMIT for batch size
        """

        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching audio segments with no SNR value")

                    # Execute the query
                    cursor.execute(select_query, (batch_size,)) # Pass the batch size as a parameter
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} records from the database.")

                    # Return the relevant fields in a dictionary format
                    return [{'segment_id': record['segment_id'], 'raw_audio_path': record['raw_audio_path'], 'processed_audio_path': record['processed_audio_path']} for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in get_audio_segments_no_snr: {ex}")
                raise

    def insert_audio_segment_snr(self, audio_segments_df: pd.DataFrame, worker_name: str) -> None:
        """
        Insert or update audio segments in bulk from a pandas DataFrame into the 'audio_segments' table.

        Parameters:
        - audio_segments_df (pd.DataFrame): DataFrame containing audio segments.

        Returns:
            None
        """
        required_columns = [
            'segment_id',
            'snr'
        ]

        # Validate DataFrame columns
        missing_columns = [col for col in required_columns if col not in audio_segments_df.columns]
        if missing_columns:
            logger.error(f"[{worker_name}] Missing required columns in the DataFrame: {missing_columns}")
            raise ValueError(f"[{worker_name}] Missing required columns in the DataFrame: {missing_columns}")
        else:
            logger.debug(f"[{worker_name}] All required columns are present in the DataFrame.")

        # Prepare data as list of tuples
        data_tuples = list(audio_segments_df[required_columns].itertuples(index=False, name=None))
        logger.debug(f"[{worker_name}] Prepared {len(data_tuples)} data tuples for insertion into 'audio_segments'.")
        logger.debug(f"[{worker_name}] {data_tuples}")

        if not data_tuples:
            logger.warning(f"[{worker_name}] No data to insert into 'audio_segments'. Exiting method.")
            return

        # Define the updte query with upsert using psycopg2's parameter placeholders (%s)
        insert_query = """
            INSERT INTO audio_segments (
                segment_id,
                snr
            )
            VALUES (%s, %s)
            ON CONFLICT (segment_id) DO UPDATE SET
                snr = EXCLUDED.snr
        """
        with self.get_db_connection(worker_namne=worker_name) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_audio_segment_snr.")

                    # Execute the bulk insert using cursor.executemany
                    cursor.executemany(insert_query, data_tuples)
                    logger.info(f"[{worker_name}] Inserted / Updated {len(data_tuples)} snr values.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_audio_segment_snr: {ex}")
                conn.rollback()
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

