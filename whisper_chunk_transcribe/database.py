# Standard Libraries
import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from psycopg2.extras import DictCursor
from sshtunnel import SSHTunnelForwarder
from typing import Any, Dict, List, Optional, Union

# Logging and Configuration
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries
import pandas as pd
import threading

# First Party Libraries
from .helper_classes import ExpSegment, ExpTestCase, TranscriptionSegment, TranscriptionWord

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

    def execute_query(self, worker_name: str, query: str, params: tuple, operation: str = 'fetch') -> Optional[Union[List[Dict], int]]:
        """
        Execute a database query.

        Args:
            worker_name (str): Name of the worker executing the query.
            query (str): SQL query to execute.
            params (tuple): Parameters for the SQL query.
            operation (str): Type of operation ('fetch', 'commit', 'execute'). Defaults to 'fetch'.

        Returns:
            Optional[Union[List[Dict], int]]: Result of the query based on the operation type.
        """
        try:
            with self.get_db_connection(worker_name) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if operation == 'fetch':
                        return cursor.fetchall()
                    elif operation == 'commit':
                        conn.commit()
                        return cursor.rowcount
                    elif operation == 'execute':
                        return cursor.rowcount
                    else:
                        logger.error(f"Unknown operation type: {operation}")
                        return None
        except psycopg2.Error as e:
            logger.error(f"SQL Error when executing query: {e}")
            if operation == 'commit':
                conn.rollback()
                logger.debug(f"[{worker_name}] Transaction rolled back due to error.")

            return None

    def set_prompt_token_length(self, prompt: str, token_length: int) -> None:
        update_query = """
            UPDATE prompt_terms
            SET tokens = %s
            WHERE term = %s;
        """
        params = (token_length, prompt)
        self.execute_query(None, update_query, params, operation='commit')

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
        params = [(term, in_game_usage_score, general_speech_score, impact_transcription_score, confusion_potential_score, tokens)
                  for term, in_game_usage_score, general_speech_score, impact_transcription_score, confusion_potential_score, tokens in data_tuples]
        self.execute_query(None, insert_query, params, operation='commit')

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
        params = [(video_id, segment_number, start_time, end_time, raw_audio_path, processed_audio_path)
                  for video_id, segment_number, start_time, end_time, raw_audio_path, processed_audio_path in data_tuples]
        self.execute_query(worker_name, insert_query, params, operation='commit')

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

        params = (experiment_id,)

        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [record['local_path'] for record in result]
        else:
            return []

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

        params = (batch_size,)

        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [{'segment_id': record['segment_id'], 'raw_audio_path': record['raw_audio_path'], 'processed_audio_path': record['processed_audio_path']} for record in result]
        else:
            return []

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
        params = [(segment_id, snr) for segment_id, snr in data_tuples]
        self.execute_query(worker_name, insert_query, params, operation='commit')

    def get_most_recent_modified_exp_id(self, worker_name: str) -> int:
        select_query = """
            SELECT experiment_id
            FROM exp_experiments
            ORDER BY modified_at DESC
            LIMIT 1;
        """
        params = ()
        return self.execute_query(worker_name, select_query, params, operation='fetch')

    def get_exp_experiment_test_cases(self, worker_name: str, experiment_id: int) -> List[ExpTestCase]:
        select_query = """
            SELECT tc.test_case_id, tc.prompt_template, tc.prompt_tokens
            FROM exp_experiment_test_cases tc
            WHERE tc.experiment_id = %s
            ORDER BY tc.test_case_id;
        """
        params = (experiment_id,)
        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [ExpTestCase(experiment_id, record['test_case_id'], record['prompt_template'], record['prompt_tokens']) for record in result]
        else:
            return []

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
        params = (experiment_id, is_dynamic)
        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [ExpSegment(record['segment_id'], record['video_id'], record['raw_audio_path'], record['processed_audio_path'], record['test_case_id']) for record in result]
        else:
            return []

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
            ON CONFLICT (prompt, prompt_tokens, experiment_test_case_id) DO UPDATE SET
                prompt = EXCLUDED.prompt
            RETURNING test_prompt_id;
        """
        params = (prompt, prompt_tokens, experiment_test_case_id)
        result = self.execute_query(worker_name, select_query, params)
        if result:
            test_prompt_id = result[0]['test_prompt_id']
            logger.info(f"[{worker_name}] Test prompt already exists with id: {test_prompt_id}")
            return test_prompt_id

        result = self.execute_query(worker_name, insert_query, params, operation='commit')
        if result:
            test_prompt_id = result[0]['test_prompt_id']
            logger.info(f"[{worker_name}] Test prompt inserted with id: {test_prompt_id}")
            return test_prompt_id
        else:
            logger.error(f"[{worker_name}] Insert operation did not return test_prompt_id.")
            raise Exception(f"[{worker_name}] Failed to insert test_prompt_id.")

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
        params = (
            test_case.experiment_id,
            segment_id,
            test_case.test_case_id,
            test_case.test_prompt_id,
            is_raw_audio,
            average_probability,
            average_avg_logprob
        )
        result = self.execute_query(worker_name, insert_query, params, operation='fetch')
        if result:
            return result[0]['test_id']
        else:
            return None

    def insert_exp_test_transcriptions(self, worker_name: str, test_id: int, transcription_text: str) -> None:
        insert_query = """
            INSERT INTO "exp_test_transcriptions" (
                test_id,
                transcription_text
            )
            VALUES (%s, %s)
            ON CONFLICT (test_id) DO NOTHING;
        """
        params = (test_id, transcription_text)
        self.execute_query(worker_name, insert_query, params, operation='commit')

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

        params = [(test_id, segment_number, segment_text, avg_logprob, start_time, end_time)
                  for test_id, segment_number, segment_text, avg_logprob, start_time, end_time in data_tuples]
        self.execute_query(worker_name, insert_query, params, operation='commit')

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

        params = [(test_id, word_number, word_text, start_time, end_time, probability)
                  for test_id, word_number, word_text, start_time, end_time, probability in data_tuples]
        self.execute_query(worker_name, insert_query, params, operation='commit')

    def get_previous_transcription(self, worker_name: str, experiment_id: int, segment_id: int) -> tuple:
        select_query = """
            WITH current_segment AS (
                SELECT 
                    cs.video_id, 
                    cs.segment_number
                FROM 
                    exp_experiment_segments ees
                JOIN 
                    audio_segments cs 
                    ON ees.segment_id = cs.segment_id
                WHERE TRUE
                    AND ees.experiment_id = %s
                    AND ees.segment_id = %s
            ),
            previous_segments AS (
                SELECT 
                    es.segment_id, 
                    es.segment_number,
                    es.raw_audio_path
                FROM 
                    audio_segments es
                JOIN 
                    current_segment cs 
                    ON es.video_id = cs.video_id
                WHERE 
                    es.segment_number < cs.segment_number
                ORDER BY 
                    es.segment_number DESC
            )
            SELECT 
                et.is_raw_audio,
                STRING_AGG(ett.transcription_text, ' ') AS aggregated_transcription_text
            FROM 
                previous_segments ps
            JOIN 
                exp_tests et 
                ON ps.segment_id = et.segment_id
            JOIN 
                exp_test_transcriptions ett 
                ON et.test_id = ett.test_id
            WHERE 
                et.deleted_at IS NULL 
                AND ett.deleted_at IS NULL
            GROUP BY
                et.is_raw_audio
            ORDER BY 
                et.is_raw_audio ASC;
        """

        params = (experiment_id, segment_id)
        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [(record['is_raw_audio'], record['aggregated_transcription_text']) for record in result]
        else:
            return []
