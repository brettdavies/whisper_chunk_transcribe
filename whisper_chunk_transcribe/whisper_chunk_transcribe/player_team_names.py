# Standard Libraries
import os
import json
import asyncio
from signal import pause
import requests
from typing import List, Optional

# Logging and Configuration
from loguru import logger
from dotenv import load_dotenv

# Third Party Libraries

# First Party Libraries
from .logger_config import LoggerConfig
from .database import DatabaseOperations
from .helper_classes import Team, Player, Game

# Load environment variables
load_dotenv()

# Configure loguru
LOGGER_NAME = "GameFetcher"
LoggerConfig.setup_logger(log_name=LOGGER_NAME, log_file_dir = "./data/log/", log_level=os.environ.get("LOG_LEVEL", "INFO"))

worker_name = LOGGER_NAME

class GameFetcher:
    async def fetch_player_data(self, espn_id: int) -> str:
        """
        Fetch player data from the ESPN API.

        Args:
            espn_id (int): The ESPN ID of the game.

        Returns:
            List[Player] or None: A list of Player objects, or None if there was an error.
        """
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={espn_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f"[{worker_name}] Player data fetched for ESPN ID {espn_id}.")
            
            # Extract only the relevant part of the JSON
            data = response.json()
            players_data = data.get('boxscore', {}).get('players', [])

            return json.dumps(players_data)
            
        except requests.RequestException as e:
            logger.error(f"[{worker_name}] Failed to fetch player data for ESPN ID {espn_id}. {e}")
            return None
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"[{worker_name}] Failed to decode JSON for ESPN ID {espn_id}. {e}")
            return None
        except Exception as e:
            logger.error(f"[{worker_name}] Error fetching player data for ESPN ID {espn_id}. {e}")
            return None
        
    async def extract_players(self, player_data_dict, home_index: int, position_index: int, team_id: int) -> List[Player]:
        players = []
        for athlete in player_data_dict[home_index]['statistics'][position_index]['athletes']:
            players.append(
                        Player(
                            player_id=athlete['athlete']['id'],      # Ensure this key exists
                            name=athlete['athlete']['displayName'],  # Ensure this key exists
                            team_id=team_id
                        )
                    )                    
        # logger.debug(f"[{worker_name}] Players: {players}")
        logger.debug(f"[{worker_name}] Players: {len(players)}")
        return players

    async def process_game(self, video_id: str, espn_id: int) -> Optional[Game]:
        """
        Process a game and extract relevant information.

        Args:
            video_id (str): The video ID.
            espn_id (int): The ESPN ID of the game.

        Returns:
            Game or None: A Game object containing the extracted information 
                          or None if any of the required fields are missing.
        """
        try:
            # Fetch data from the ESPN API using the espn_id
            player_data = await self.fetch_player_data(espn_id)

            # Parse the JSON string into a Python object (dictionary or list)
            player_data_dict = json.loads(player_data)
        
            if player_data:
                logger.debug(f"[{worker_name}] Player data fetched for video_id {video_id} and espn_id {espn_id}.")
                # Extract team IDs and display names
                home_team_data = player_data_dict[0]['team']
                away_team_data = player_data_dict[1]['team']

                home_team = Team(
                    team_id=home_team_data['id'],
                    display_name=home_team_data['displayName'],
                    abbreviation=home_team_data['abbreviation']
                )
                # logger.debug(f"[{worker_name}] Home team: {home_team}")

                away_team = Team(
                    team_id=away_team_data['id'],
                    display_name=away_team_data['displayName'],
                    abbreviation=away_team_data['abbreviation']
                )
                # logger.debug(f"[{worker_name}] Away team: {away_team}")

                # Extract Home Players
                home_players = []
                home_players.extend(await self.extract_players(player_data_dict, 0, 0, home_team.team_id)) # Home Fielders
                home_players.extend(await self.extract_players(player_data_dict, 0, 1, home_team.team_id)) # Home Pitchers

                # Extract Away Players
                away_players = []
                away_players.extend(await self.extract_players(player_data_dict, 1, 0, away_team.team_id)) # Away Fielders
                away_players.extend(await self.extract_players(player_data_dict, 1, 1, away_team.team_id)) # Away Pitchers

                # Create and return a Game instance
                game = Game(
                    video_id=video_id,
                    espn_id=espn_id,
                    home_team=home_team,
                    away_team=away_team,
                    home_players=home_players,
                    away_players=away_players
                )
                logger.debug(f"[{worker_name}] Game instance: {game}")

                return game
            else:
                return None

        except KeyError as e:
            logger.error(f"[{worker_name}] process_game() KeyError: {e}")
            return None

async def main():
    # Initialize the DatabaseOperations class
    db_ops = DatabaseOperations()

    # Fetch video IDs and ESPN IDs from the database
    ids = db_ops.get_exp_video_espn_map("GameFetcher")
    # logger.debug(f"Video and ESPN IDs fetched: {ids}")

    for video_id, espn_id in ids:
        try:
            game_fetcher = GameFetcher()
            game_instance = await game_fetcher.process_game(video_id, espn_id)
            # logger.debug(f"[{worker_name}] Home players: {len(game_instance.home_players)}")
            # logger.debug(f"[{worker_name}] Away players: {len(game_instance.away_players)}")
            
            if game_instance:
                # Upsert the game data into the database
                logger.debug(f"[{worker_name}] Upserting game data for video_id {video_id} and espn_id {espn_id}.")
                db_ops.upsert_game_data(worker_name, game_instance)
            else:
                logger.warning(f"[{worker_name}] No game data found for video_id {video_id} and espn_id {espn_id}.")
        except Exception as e:
            logger.error(f"[{worker_name}] Error processing game with video_id {video_id} and espn_id {espn_id}: {e}")
    
    # Close the database connection
    db_ops.close()

if __name__ == "__main__":
    # Run the main function with the provided prompt
    asyncio.run(main())

# Run the script as a Module:
# python -m whisper_chunk_transcribe.player_team_names
