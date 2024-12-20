# Standard Libraries
import re
import pytz
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Logging
from loguru import logger

# First Party Libraries
from .espn_metadata import Metadata
from .database import DatabaseOperations
from .utils import Utils

async def populate_event_metadata_queue(queue_manager: QueueManager) -> None:
    """
    Populates the event metadata queue with dates that require event metadata processing.

    Args:
        queue_manager (QueueManager): The QueueManager instance to add dates to.

    Returns:
        None
    """
    event_dates = DatabaseOperations.get_dates_no_event_metadata()
    for date_str in event_dates:
        await queue_manager.event_metadata_queue.put(date_str)
    logger.info(f"Size of event_metadata_queue: {queue_manager.event_metadata_queue.qsize()}")


async def worker_event_metadata(queue_manager: QueueManager, worker_id: str) -> None:
    """
    Processes event metadata by retrieving dates without event metadata and running the EventFetcher.

    Args:
        queue_manager (QueueManager): Manages the queues and active tasks.
        worker_id (str): Identifier for the worker instance.

    Returns:
        None

    Raises:
        Exception: If an error occurs during event metadata processing.
    """
    while True:
        try:
            queue_manager.active_tasks['event_metadata'] += 1

            if queue_manager.event_metadata_queue.qsize() > 0:
                date_obj = await asyncio.wait_for(queue_manager.event_metadata_queue.get(), timeout=1)
                logger.info(f"[Worker {worker_id}] Size of event_metadata_queue after get: {queue_manager.event_metadata_queue.qsize()}")

                # Assuming date_str is a datetime.date object
                date_str = date_obj.strftime('%Y%m%d')

                # Initialize the EventFetcher with the date_str
                event_fetcher = EventFetcher()

                # Run the EventFetcher
                logger.info(f"[Worker {worker_id}] Running EventFetcher for date: {date_str}")
                await event_fetcher.run(date_str)

                queue_manager.event_metadata_queue.task_done()
        except asyncio.TimeoutError:
            logger.error(f"[Worker {worker_id}] asyncio.TimeoutError")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error saving metadata: {e}")
        finally:
            queue_manager.active_tasks['event_metadata'] -= 1
            if queue_manager.event_metadata_queue.empty():
                break  # Exit only when the queue is empty
        await asyncio.sleep(.250)
    logger.info(f"[Worker {worker_id}] Save metadata worker has finished.")



class EventFetcher:
    """
    Fetches and processes event data from the ESPN API.

    The `EventFetcher` retrieves Major League Baseball event data for a specified date,
    processes the data, and saves it to the database.

    Attributes:
        team_abbreviations: A dictionary mapping team abbreviations to team names.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    LIMIT = 1000

    def __init__(self) -> None:
        """
        Initializes the EventFetcher with team abbreviations.
        """
        self.team_abbreviations = Metadata.team_abbreviations

    async def setup(self, date_stub: str) -> None:
        """
        Sets up the EventFetcher with the specified date.

        Normalizes the date, constructs the API URL, and checks if data for the date is already loaded.

        Args:
            date_stub: A string representing the date in various formats (e.g., '20210101', '2021-01-01').

        Raises:
            ValueError: If the date_stub is not in a valid format.
        """
        self.date_stub = await Utils.normalize_date_stub(date_stub)
        self.url = f"{self.BASE_URL}?limit={self.LIMIT}&dates={self.date_stub}"
        logger.debug(f"Setting up EventFetcher with date_stub: {self.date_stub}")
        self.already_loaded = DatabaseOperations.check_if_existing_e_events_by_date(self.date_stub)
        logger.debug(f"Data already loaded: {self.already_loaded}")

    async def fetch_data(self) -> Optional[Dict]:
        """
        Fetches event data from the ESPN API.

        Returns:
            A dictionary containing the fetched data if successful, or None if the request fails.
        """
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None

    async def process_event(self, event) -> Optional[List]:
        """
        Processes a single event and extracts relevant information.

        Args:
            event: A dictionary containing event data.

        Returns:
            A list containing extracted information:
            [event_id, ny_time, season_type, short_name, home_team, away_team,
            home_team_normalized, away_team_normalized], or None if any required fields are missing.
        """
        try:
            event_id = event.get('id')
            date = event.get('date')
            short_name = event.get('shortName')
            season_type = event.get('season', {}).get('type')
            # Extract the home team abbreviation
            home_team = next((team['team']['abbreviation'] for team in event.get('competitions', [{}])[0].get('competitors', []) if team.get('homeAway') == 'home'), None)
            # Extract the away team abbreviation
            away_team = next((team['team']['abbreviation'] for team in event.get('competitions', [{}])[0].get('competitors', []) if team.get('homeAway') == 'away'), None)
            
            if not all([event_id, date, short_name, season_type, home_team, away_team]):
                return None
            
            else:
                # Normalize the team abbrevations
                home_team_normalized, away_team_normalized = Utils.extract_teams(f"{away_team} @ {home_team}")
            
                date_no_z = date.rstrip('Z')
                utc_time = datetime.strptime(date_no_z, '%Y-%m-%dT%H:%M')
                utc_time = pytz.utc.localize(utc_time)
                ny_time = utc_time.astimezone(pytz.timezone('America/New_York'))
                return [event_id, ny_time, season_type, short_name, home_team, away_team, home_team_normalized, away_team_normalized]
        except KeyError as e:
            logger.error(f"process_event() KeyError: {e}")
            return None

    async def extract_events(self, data) -> List[List]:
        """
        Extracts and processes events from the provided data.

        Args:
            data: A dictionary containing events data.

        Returns:
            A list of processed events, each represented as a list of extracted information.
        """
        events = data.get('events', [])
        processed_events = []
        for event in events:
            processed_event = await self.process_event(event)
            if processed_event:
                processed_events.append(processed_event)
        return processed_events

    async def create_dataframe(self, events_data) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from the given events data.

        Args:
            events_data: A list of event data, where each event is represented as a list.

        Returns:
            A pandas DataFrame containing the events data with specified columns.
        """
        columns = ['event_id', 'date', 'type', 'short_name', 'home_team','away_team', 'home_team_normalized', 'away_team_normalized']
        return pd.DataFrame(events_data, columns=columns)

    async def save_to_database(self, dataframe) -> None:
        """
        Saves the events data DataFrame to the database.

        Args:
            dataframe: A pandas DataFrame containing events data to be saved.
        """
        DatabaseOperations.save_events(dataframe)

    async def run(self, date_stub:str) -> None:
        """
        Runs the EventFetcher to fetch, process, and save events data for a given date.

        Args:
            date_stub: A string representing the date in various formats.

        Raises:
            ValueError: If the date_stub is not in a valid format.
        """
        await self.setup(date_stub)
        logger.info(f"Starting EventFetcher with date_stub: {self.date_stub}")
        if not self.already_loaded:
            data = await self.fetch_data()
            if data:
                events_data = await self.extract_events(data)
                if events_data:
                    df = await self.create_dataframe(events_data)
                    await self.save_to_database(df)
                    logger.info("Data successfully saved to the database.")
                else:
                    logger.warning("No events data to process.")
            else:
                logger.error("Failed to fetch data.")
        else:
            logger.info("Data already loaded for the given date_stub.")
