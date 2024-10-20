# Standard Libraries
import datetime
from typing import List, Optional, Tuple

# Logging
from loguru import logger

# Third Party Libraries
import pandas as pd

# First Party Libraries
from .helper_classes import Game
from .database import DatabaseOperations

class ExtendedDatabaseOperations(DatabaseOperations):
    def get_dates_no_event_metadata(self) -> List[datetime.datetime]:
        """
        Retrieves dates that have no associated event metadata.

        Returns:
            A list of dates lacking event metadata.
        """
        query = """
        SELECT *
        FROM (
            SELECT DISTINCT yt_metadata.event_date_local_time
            FROM yt_metadata
            LEFT JOIN e_events ON yt_metadata.event_date_local_time = e_events.date::DATE
            WHERE e_events.date IS NULL
        ) AS distinct_dates
        ORDER BY RANDOM()
        """
        result = self.execute_query(None, query, ())
        if result is not None:
            dates_no_event_metadata = [row[0] for row in result]
        else:
            dates_no_event_metadata = []
        return dates_no_event_metadata

    def save_events(self, df: pd.DataFrame) -> None:
        """
        Inserts events into the 'e_events' table in the database.

        Args:
            df: The DataFrame containing the events data.

        Raises:
            ValueError: If the DataFrame does not contain all the required columns.
        """
        required_columns = ['event_id', 'date', 'type', 'short_name', 'home_team', 'away_team',
                            'home_team_normalized', 'away_team_normalized']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' in the DataFrame.")
                raise ValueError(f"Missing required column '{col}' in the DataFrame.")

        records = df[required_columns].values.tolist()

        insert_query = """
        INSERT INTO e_events (event_id, date, type, short_name, home_team, away_team, home_team_normalized, away_team_normalized)
        VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (event_id) DO NOTHING
        """

        params = records
        self.execute_query(None, insert_query, params, operation='commit')

    def check_if_existing_e_events_by_date(self, date_obj: datetime.datetime) -> bool:
        """
        Checks if there are existing events in 'e_events' for the given date.

        Args:
            date_obj: The date to check for existing events.

        Returns:
            True if events exist for the given date, False otherwise.
        """
        query = """
        SELECT COUNT(1)
        FROM e_events
        WHERE (date AT TIME ZONE 'America/New_York')::date = %s::date
        """
        params = (date_obj,)
        count_result = self.execute_query(None, query, params)[0][0]

        return count_result > 0

    def get_e_events_team_info(self, date_obj: datetime.datetime, opposing_team: str, is_home_unknown: bool) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves normalized team abbreviation from 'e_events'.

        Args:
            date_obj: The date of the event.
            opposing_team: The name of the opposing team.
            is_home_unknown: Flag indicating whether the home team is unknown.

        Returns:
            A tuple containing the event ID (optional) and the normalized team abbreviation (optional).
        """
        event_id = None
        team = 'Unknown'
        team_column = 'home_team_normalized' if is_home_unknown else 'away_team_normalized'
        opposing_column = 'away_team' if is_home_unknown else 'home_team'

        select_query = f"""
        SELECT {team_column}
        FROM e_events
        WHERE (date AT TIME ZONE 'America/New_York')::date = %s::date
            AND {opposing_column} = %s
        """

        params = (date_obj, opposing_team)
        result = self.execute_query(None, select_query, params)
        if result:
            team = result[0][0]
        else:
            team = 'Unknown'
        return event_id, team

    def get_event_id(self, date_obj: datetime.datetime, home_team: str, away_team: str) -> Optional[str]:
        """
        Retrieves the event ID based on date and team information.

        Args:
            date_obj: The date of the event.
            home_team: The home team name.
            away_team: The away team name.

        Returns:
            The event ID if found, otherwise None.
        """
        if home_team == 'Unknown' and away_team == 'Unknown':
            return None

        if home_team != 'Unknown' and away_team != 'Unknown':
            select_query = """
            SELECT event_id
            FROM e_events
            WHERE (date AT TIME ZONE 'America/New_York')::date = %s::date
                AND home_team_normalized = %s
                AND away_team_normalized = %s
            """
            params = (date_obj, home_team, away_team)
        else:
            team_column = 'home_team_normalized' if home_team != 'Unknown' else 'away_team_normalized'
            team_value = home_team if home_team != 'Unknown' else away_team
            select_query = f"""
            SELECT event_id
            FROM e_events
            WHERE (date AT TIME ZONE 'America/New_York')::date = %s::date
                AND {team_column} = %s
            """
            params = (date_obj, team_value)
        
        result = self.execute_query(None, select_query, params)
        event_id = result[0] if result else None
        return event_id

    def get_exp_video_espn_map(self, worker_name: str) -> tuple:
        select_query = """
            SELECT
                map.yt_id,
                map.espn_id
            FROM exp_experiment_videos AS exp_vid
            JOIN video_espn_mapping AS map ON map.yt_id = exp_vid.video_id;
        """

        params = ()
        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [(record['yt_id'], record['espn_id']) for record in result]
        else:
            return []

    def upsert_game_data(self, worker_name: str, game: Game) -> None:
        # Prepare bulk upsert for teams using a dictionary to keep the last occurrence
        teams_data = {
            (team.team_id, team.display_name): team.abbreviation  # Use a tuple of (team_id, display_name) as the key
            for team in game.get_upsert_teams()
        }

        # Convert back to a list of tuples
        teams_data = [(team_id, display_name, abbreviation) for (team_id, display_name), abbreviation in teams_data.items()]
        
        # Upsert Teams in bulk
        if teams_data:
            teams_query = """
                INSERT INTO e_teams (team_id, display_name, abbreviation)
                VALUES %s
                ON CONFLICT (team_id) 
                DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    abbreviation = EXCLUDED.abbreviation;
            """
            self.execute_query(worker_name, teams_query, teams_data, operation='commit')
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
            players_query = """
                INSERT INTO e_players (player_id, name)
                VALUES %s
                ON CONFLICT (player_id) 
                DO UPDATE SET
                    name = EXCLUDED.name;
            """
            self.execute_query(worker_name, players_query, players_data, operation='commit')
            logger.debug(f"[{worker_name}] Upserted {len(players_data)} players.")

        # Insert Game Record
        game_query = """
            INSERT INTO e_games (video_id, espn_id, home_team_id, away_team_id)
            VALUES %s
            RETURNING game_id;
        """
        game_id = self.execute_query(worker_name, game_query, [(game.video_id, game.espn_id, game.home_team.team_id, game.away_team.team_id)], operation='fetch')[0]['game_id']
        
        logger.debug(f"[{worker_name}] Inserted game record with ID: {game_id}")

        # Upsert Game Players in bulk using a dictionary to keep the last occurrence
        game_players_dict = {
            (player.player_id, player.team_id): game_id  # Use a tuple of (player_id, team_id) as the key
            for player in game.get_upsert_players()
        }

        # Convert back to a list of tuples
        game_players_data = [(game_id, player_id, team_id) for (player_id, team_id), game_id in game_players_dict.items()]

        if game_players_data:
            game_players_query = """
                INSERT INTO e_game_players (game_id, player_id, team_id)
                VALUES %s
                ON CONFLICT (game_id, player_id, team_id) 
                DO NOTHING;
            """
            self.execute_query(worker_name, game_players_query, game_players_data, operation='commit')
            logger.debug(f"[{worker_name}] Upserted {len(game_players_data)} game players.")

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

        params = (video_id,)

        result = self.execute_query(worker_name, select_query, params)
        if result:
            return [(record['team_name'], record['players']) for record in result]
        else:
            return []
