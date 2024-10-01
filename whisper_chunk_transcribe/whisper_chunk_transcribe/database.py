# Standard Libraries
import os
import psycopg2
from psycopg2 import pool, extras
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from psycopg2.extras import DictCursor
from sshtunnel import SSHTunnelForwarder

# Logging and Configuration
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
import pandas as pd
import threading

# First Party Libraries
from .helper_classes import ExpSegment, ExpTestCase, TranscriptionSegment, TranscriptionWord, Game

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
        missing = [var for var in required_vars if not os.environ.get(var)]
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
    def get_db_connection(self, worker_name: Optional[str]) -> Any:
        """
        Context manager for database connection.
        Ensures that the connection is properly returned to the pool.
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            if conn is None:
                logger.error(f"{"["+worker_name+"] " if worker_name else None}Failed to obtain database connection from the pool.")
                raise psycopg2.OperationalError(f"{"["+worker_name+"] " if worker_name else None}Unable to obtain database connection.")
            yield conn
        except psycopg2.Error as e:
            logger.error(f"{"["+worker_name+"] " if worker_name else None}Error obtaining database connection: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
                logger.debug(f"{"["+worker_name+"] " if worker_name else None}Database connection returned to the pool.")

    def get_prompt_terms(self) -> List[str]:
        select_query = """
            SELECT term FROM prompt_terms;
        """
        with self.get_db_connection(None) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug("Fetching prompt terms from the database.")

                    # Execute the query
                    cursor.execute(select_query)
                    records = cursor.fetchall()
                    logger.debug(f"Retrieved {len(records)} prompt term records from the database.")

                    # Extract 'term' from each record
                    return [record[0] for record in records]

            except Exception as ex:
                logger.error(f"Error in get_prompt_terms: {ex}")
                raise

    def set_prompt_token_length(self, prompt: str, token_length: int) -> None:
        update_query = """
            UPDATE prompt_terms
            SET tokens = %s
            WHERE term = %s;
        """
        with self.get_db_connection(None) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"Setting token length for prompt: {prompt}")

                    # Execute the query
                    cursor.execute(update_query, (token_length, prompt))
                    conn.commit()
                    logger.info(f"Token length set for prompt: {prompt}")

            except Exception as ex:
                logger.error(f"Error in set_prompt_token_length: {ex}")
                conn.rollback()
                logger.debug("Transaction rolled back due to error.")
                raise

    def get_exp_experiment_prompt_terms(self, experiment_id: int) -> List[str]:
        select_query = """
            SELECT term
            FROM exp_experiment_prompt_terms
            WHERE experiment_id = %s
            ORDER BY final_score DESC;
        """
        with self.get_db_connection(None) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug("Fetching prompt terms from the database.")

                    # Execute the query
                    cursor.execute(select_query, (experiment_id,))
                    records = cursor.fetchall()
                    logger.debug(f"Retrieved {len(records)} records from the database.")

                    # Extract 'term' from each record
                    return [record[0] for record in records]

            except Exception as ex:
                logger.error(f"Error in get_prompt_terms: {ex}")
                raise

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
        with self.get_db_connection(worker_name) as conn:
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

        with self.get_db_connection(worker_name) as conn:
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
        with self.get_db_connection(worker_name) as conn:
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

    def get_most_recent_modified_exp_id(self, worker_name: str) -> int:
        select_query = """
            SELECT experiment_id
            FROM exp_experiments
            ORDER BY modified_at DESC
            LIMIT 1;
        """
        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching most recently modified experiment ID")

                    # Execute the query
                    cursor.execute(select_query)
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} records from the database.")

                    # Return the relevant fields in a dictionary format
                    return records[0]['experiment_id']

            except Exception as ex:
                logger.error(f"[{worker_name}] SQL Error in get_most_recent_modified_exp_id: {ex}")
                raise

    def get_exp_experiment_test_cases(self, worker_name: str, experiment_id: int) -> List[ExpTestCase]:
        select_query = """
            SELECT tc.test_case_id, tc.prompt_template, tc.prompt_tokens
            FROM exp_experiment_test_cases tc
            WHERE tc.experiment_id = %s
            ORDER BY tc.test_case_id;
        """
        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching test cases for experiment ID: {experiment_id}")

                    # Execute the query
                    cursor.execute(select_query, (experiment_id,))
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} test cases from the database for experiment {experiment_id}.")

                    # Return the relevant fields in a dictionary format
                    return [ExpTestCase(experiment_id, record['test_case_id'], record['prompt_template'], record['prompt_tokens']) for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] SQL Error in get_exp_experiment_test_cases: {ex}")
                raise

    def get_exp_experiment_segments(self, worker_name: str, experiment_id: int, is_dynamic: bool) -> List[ExpSegment]:
        select_query = """
            SELECT seg.segment_id, seg.video_id, seg.raw_audio_path, seg.processed_audio_path, exp_seg.test_case_id
            FROM exp_experiment_segments AS exp_seg
            JOIN audio_segments AS seg ON seg.segment_id = exp_seg.segment_id
            JOIN exp_experiment_test_cases AS exp_tc ON exp_tc.test_case_id = exp_seg.test_case_id
            JOIN e_games AS game ON game.video_id = seg.video_id
            WHERE TRUE
                AND exp_seg.experiment_id = %s
                AND exp_tc.use_prev_transcription = %s
            ORDER BY seg.segment_id ASC;
        """
        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching test segments for experiment ID: {experiment_id}")

                    # Execute the query
                    cursor.execute(select_query, (experiment_id, is_dynamic))
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} test segment records from the database.")

                    # Return the relevant fields in a dictionary format
                    return [ExpSegment(record['segment_id'], record['video_id'], record['raw_audio_path'], record['processed_audio_path'], record['test_case_id']) for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] SQL Error in get_exp_experiment_segments: {ex}")
                raise

    def insert_test_prompt(self, worker_name: str, prompt: str, prompt_tokens: int, experiment_test_case_id: int) -> int:
        select_query = """
            SELECT test_prompt_id
            FROM exp_test_prompts
            WHERE (prompt IS NOT DISTINCT FROM %s) AND prompt_tokens = %s AND experiment_test_case_id = %s AND deleted_at IS NULL
            LIMIT 1;
        """
        insert_query = """
            INSERT INTO exp_test_prompts (prompt, prompt_tokens, experiment_test_case_id)
            VALUES (%s, %s, %s)
            RETURNING test_prompt_id;
        """
        with self.get_db_connection(worker_name=worker_name) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_test_prompt.")

                    # 1. Check for Existing Record
                    cursor.execute(select_query, (prompt, prompt_tokens, experiment_test_case_id))
                    result = cursor.fetchone()

                    if result:
                        test_prompt_id = result[0]
                        logger.info(f"[{worker_name}] Test prompt already exists with id: {test_prompt_id}")
                        return test_prompt_id

                    # 2. Insert New Record
                    cursor.execute(insert_query, (prompt, prompt_tokens, experiment_test_case_id))
                    insert_result = cursor.fetchone()

                    if insert_result:
                        test_prompt_id = insert_result[0]
                        logger.info(f"[{worker_name}] Test prompt inserted with id: {test_prompt_id}")
                        conn.commit()
                        logger.debug(f"[{worker_name}] Transaction committed successfully.")
                        return test_prompt_id
                    else:
                        logger.error(f"[{worker_name}] Insert operation did not return test_prompt_id.")
                        conn.rollback()
                        logger.debug(f"[{worker_name}] Transaction rolled back due to missing test_prompt_id.")
                        raise Exception(f"[{worker_name}] Failed to insert test_prompt_id.")

            except psycopg2.IntegrityError as ie:
                # Handle race condition: another process might have inserted the same record
                conn.rollback()
                logger.warning(f"[{worker_name}] IntegrityError occurred: {ie}. Retrying SELECT.")
                try:
                    with conn.cursor() as retry_cursor:
                        retry_cursor.execute(select_query, (prompt, prompt_tokens, experiment_test_case_id))
                        retry_result = retry_cursor.fetchone()
                        if retry_result:
                            test_prompt_id = retry_result[0]
                            logger.info(f"Test prompt already exists after IntegrityError with id: {test_prompt_id}")
                            return test_prompt_id
                        else:
                            logger.error(f"[{worker_name}] Retry SELECT did not find the test_prompt_id.")
                            raise Exception("Failed to insert or retrieve test_prompt_id after IntegrityError.")
                except Exception as ex_inner:
                    logger.error(f"[{worker_name}] Error during retry SELECT: {ex_inner}")
                    raise
            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_test_prompt: {ex}")
                conn.rollback()
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

    def insert_exp_test(self, worker_name: str, test_case: ExpTestCase, segment_id: int, is_raw_audio: bool, average_probability: float, average_avg_logprob: float) -> int:
        insert_query = """
            INSERT INTO "exp_tests" (
                experiment_id,
                segment_id,
                test_case_id,
                test_prompt_id,
                is_raw_audio,
                average_probability,
                average_avg_logprob
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, segment_id, test_case_id, test_prompt_id, is_raw_audio) DO UPDATE SET
                average_probability = EXCLUDED.average_probability,
                average_avg_logprob = EXCLUDED.average_avg_logprob
            RETURNING test_id;
        """
        with self.get_db_connection(worker_name) as conn:
            test_id = None
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_exp_test.")

                    cursor.execute(insert_query, (
                        test_case.experiment_id,
                        segment_id,
                        test_case.test_case_id,
                        test_case.test_prompt_id,
                        is_raw_audio,
                        average_probability,
                        average_avg_logprob
                    ))
                    test_id = cursor.fetchone()[0]
                    logger.debug(f"[{worker_name}] Inserted / Updated Test ID: {test_id}.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_exp_test: {ex}")
                conn.rollback()
                test_id = None
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

            finally:
                return test_id

    def insert_exp_test_transcriptions(self, worker_name: str, test_id: int, transcription_text: str) -> None:
        insert_query = """
            INSERT INTO "exp_test_transcriptions" (
                test_id,
                transcription_text
            )
            VALUES (%s, %s)
            ON CONFLICT (test_id) DO NOTHING;
        """
        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_exp_test_transcriptions.")

                    cursor.execute(insert_query, (
                        test_id,
                        transcription_text
                    ))
                    logger.debug(f"[{worker_name}] Inserted test transcript: {test_id}.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_exp_test_transcriptions: {ex}")
                conn.rollback()
                test_id = None
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

    def insert_exp_test_transcription_segments(self, worker_name: str, test_id: int, segments: List[TranscriptionSegment]) -> None:
        insert_query = """
            INSERT INTO "exp_test_transcription_segments" (
                test_id,
                segment_number,
                segment_text,
                avg_logprob,
                start_time,
                end_time
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (test_id, segment_number) DO NOTHING;
        """
        # Prepare data as list of tuples with segment_number
        data_tuples = [
            (test_id, index + 1, segment.transcribed_text, segment.avg_logprob, segment.start, segment.end)
            for index, segment in enumerate(segments)
        ]
        logger.debug(f"[{worker_name}] Prepared {len(data_tuples)} data tuples for insertion into 'exp_test_transcription_segments'.")
        logger.debug(f"[{worker_name}] {data_tuples}")

        if not data_tuples:
            logger.warning(f"[{worker_name}] No data to insert into 'exp_test_transcription_segments'. Exiting method.")
            return

        with self.get_db_connection(worker_name) as conn:
            test_id = None
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_exp_test_transcription_segments.")

                    # Execute the bulk insert using cursor.executemany
                    cursor.executemany(insert_query, data_tuples)
                    logger.info(f"[{worker_name}] Inserted transcription segment values.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_exp_test_transcription_segments: {ex}")
                conn.rollback()
                test_id = None
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

            finally:
                return test_id

    def insert_exp_test_transcription_words(self, worker_name: str, test_id: int, words: List[TranscriptionWord]) -> None:
        insert_query = """
            INSERT INTO "exp_test_transcription_words" (
                test_id,
                word_number,
                word_text,
                start_time,
                end_time,
                probability
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (test_id, word_number) DO NOTHING;
        """
        # Prepare data as list of tuples with word_number
        data_tuples = [
            (test_id, index + 1, word.word, word.start, word.end, word.probability)
            for index, word in enumerate(words)
        ]

        logger.debug(f"[{worker_name}] Prepared {len(data_tuples)} data tuples for insertion into 'exp_test_transcription_words'.")
        logger.debug(f"[{worker_name}] {data_tuples}")

        if not data_tuples:
            logger.warning(f"[{worker_name}] No data to insert into 'exp_test_transcription_words'. Exiting method.")
            return

        with self.get_db_connection(worker_name) as conn:
            test_id = None
            try:
                with conn.cursor() as cursor:
                    logger.debug(f"[{worker_name}] Connection acquired successfully for insert_exp_test_transcription_words.")

                    # Execute the bulk insert using cursor.executemany
                    cursor.executemany(insert_query, data_tuples)
                    logger.info(f"[{worker_name}] Inserted transcription segment values.")

                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"[{worker_name}] Transaction committed successfully.")

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in insert_exp_test_transcription_words: {ex}")
                conn.rollback()
                test_id = None
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")
                raise

            finally:
                return test_id

    def get_exp_video_espn_map(self, worker_name: str) -> tuple:
        select_query = """
            SELECT
                map.yt_id,
                map.espn_id
            FROM exp_experiment_videos AS exp_vid
            JOIN video_espn_mapping AS map ON map.yt_id = exp_vid.video_id;
        """

        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching audio segments with no SNR value")

                    # Execute the query
                    cursor.execute(select_query)
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} records from the database.")

                    # Return the relevant fields as a list of tuples
                    return [(record['yt_id'], record['espn_id']) for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in get_audio_segments_no_snr: {ex}")
                raise  

    def upsert_game_data(self, worker_name: str, game: Game) -> None:
        # Connect to the database
        with self.get_db_connection(worker_name) as conn:
            try:
                # Create a cursor object
                cursor = conn.cursor()

                # Prepare bulk upsert for teams using a dictionary to keep the last occurrence
                teams_data = {
                    (team.team_id, team.display_name): team.abbreviation  # Use a tuple of (team_id, display_name) as the key
                    for team in game.get_upsert_teams()
                }

                # Convert back to a list of tuples
                teams_data = [(team_id, display_name, abbreviation) for (team_id, display_name), abbreviation in teams_data.items()]
                
                # Upsert Teams in bulk
                if teams_data:
                    extras.execute_values(
                        cursor,
                        """
                        INSERT INTO e_teams (team_id, display_name, abbreviation)
                        VALUES %s
                        ON CONFLICT (team_id) 
                        DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            abbreviation = EXCLUDED.abbreviation;
                        """,
                        teams_data
                    )
                    logger.debug(f"[{worker_name}] Upserted {len(teams_data)} teams.")

                # Prepare bulk upsert for players using a dictionary to keep the last occurrence
                players_data_dict = {
                    player.player_id: player.name  # Use player_id as key and player.name as value
                    for player in game.get_upsert_players()
                }

                # Convert back to a list of tuples
                players_data = [(player_id, name) for player_id, name in players_data_dict.items()]

                # Upsert Players in bulk
                if players_data:
                    extras.execute_values(
                        cursor,
                        """
                        INSERT INTO e_players (player_id, name)
                        VALUES %s
                        ON CONFLICT (player_id) 
                        DO UPDATE SET
                            name = EXCLUDED.name;
                        """,
                        players_data
                    )
                    logger.debug(f"[{worker_name}] Upserted {len(players_data)} players.")

                # Insert Game Record
                extras.execute_values(
                    cursor,
                    """
                    INSERT INTO e_games (video_id, espn_id, home_team_id, away_team_id)
                    VALUES %s
                    RETURNING game_id;
                    """,
                    [(game.video_id, game.espn_id, game.home_team.team_id, game.away_team.team_id)]  # Wrap the tuple in a list
                )
                
                # Get the newly created game_id
                game_id = cursor.fetchone()[0]
                logger.debug(f"[{worker_name}] Inserted game record with ID: {game_id}")

                # Upsert Game Players in bulk using a dictionary to keep the last occurrence
                game_players_dict = {
                    (player.player_id, player.team_id): game_id  # Use a tuple of (player_id, team_id) as the key
                    for player in game.get_upsert_players()
                }

                # Convert back to a list of tuples
                game_players_data = [(game_id, player_id, team_id) for (player_id, team_id), game_id in game_players_dict.items()]

                if game_players_data:
                    extras.execute_values(
                        cursor,
                        """
                        INSERT INTO e_game_players (game_id, player_id, team_id)
                        VALUES %s
                        ON CONFLICT (game_id, player_id, team_id) 
                        DO NOTHING;
                        """,
                        game_players_data
                    )
                    logger.debug(f"[{worker_name}] Upserted {len(game_players_data)} game players.")

                # Commit the transaction
                conn.commit()
            except psycopg2.Error as e:
                logger.error(f"SQL Error when writing data for game {game.video_id}:\n{e}")
                conn.rollback()
                raise

    def get_teams_players(self, worker_name: str, video_id: str) -> tuple:
        select_query = """
            WITH teams_players AS (
                SELECT 
                    t.display_name AS team_name,
                    p.name AS player_name,
                    g.game_id
                FROM 
                    e_games g
                JOIN 
                    e_teams t ON t.team_id IN (g.home_team_id, g.away_team_id)
                JOIN 
                    e_game_players gp ON gp.game_id = g.game_id
                    AND gp.team_id = t.team_id
                JOIN 
                    e_players p ON p.player_id = gp.player_id
                WHERE 
                    g.video_id = %s
            )
            SELECT 
                team_name, 
                STRING_AGG(player_name, ', ') AS players
            FROM 
                teams_players
            GROUP BY 
                team_name;
        """

        with self.get_db_connection(worker_name) as conn:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    logger.debug(f"[{worker_name}] Fetching teams and players for video ID: {video_id}")

                    # Execute the query
                    cursor.execute(select_query, (video_id,))
                    records = cursor.fetchall()
                    logger.debug(f"[{worker_name}] Retrieved {len(records)} teams and player records from the database.")

                    # Return the relevant fields as a list of tuples
                    return [( record['team_name'], record['players']) for record in records]

            except Exception as ex:
                logger.error(f"[{worker_name}] Error in get_teams_players: {ex}")
                raise  
