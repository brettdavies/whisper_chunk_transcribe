-- ================================================
-- Database Schema for ESPN-related Data Storage
-- ================================================

-- ================================================
-- 1. Trigger Functions
-- ================================================

-- 1.1. Trigger Function to Update modified_at Column
CREATE OR REPLACE FUNCTION update_modified_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 1.2. Trigger Function to Manage Video ESPN Mapping
CREATE OR REPLACE FUNCTION manage_video_espn_mapping()
RETURNS TRIGGER AS $$
DECLARE
    espn_id INT;
    existing_record RECORD;
BEGIN
    -- Extract espn_id from the local_path
    SELECT (REGEXP_MATCHES(NEW.local_path, '{e-(\d+)}'))[1]::INT INTO espn_id
    WHERE NEW.local_path ILIKE '%{e-%';

    -- Check for an existing record
    SELECT * INTO existing_record 
    FROM video_espn_mapping 
    WHERE yt_id = NEW.video_id AND espn_id = espn_id;

    IF NOT FOUND THEN
        -- If no existing record, insert a new one
        INSERT INTO video_espn_mapping (yt_id, espn_id, created_at, modified_at)
        VALUES (NEW.video_id, espn_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
    ELSIF existing_record IS NOT NULL THEN
        IF existing_record.espn_id = espn_id THEN
            -- If the existing record and new record are the same, update modified_at
            UPDATE video_espn_mapping
            SET modified_at = CURRENT_TIMESTAMP
            WHERE yt_id = NEW.video_id AND espn_id = espn_id;
        ELSE
            -- If the existing and new records differ, soft delete the existing record
            UPDATE video_espn_mapping
            SET deleted_at = CURRENT_TIMESTAMP
            WHERE yt_id = NEW.video_id AND espn_id = existing_record.espn_id;

            -- Insert the new record
            INSERT INTO video_espn_mapping (yt_id, espn_id, created_at, modified_at)
            VALUES (NEW.video_id, espn_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 1.3. Trigger Function to Propagate Soft Deletes from e_events to related tables
CREATE OR REPLACE FUNCTION propagate_soft_delete_e_events()
RETURNS TRIGGER AS $$
DECLARE
    tables_with_event_id TEXT[] := ARRAY[
        'e_games',
        'video_espn_mapping'
    ];
    tables_with_game_id TEXT[] := ARRAY[
        'e_game_players'
    ];
    table_name TEXT;
BEGIN
    IF NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL THEN
        -- Soft delete records in tables that reference e_events.event_id
        FOREACH table_name IN ARRAY tables_with_event_id LOOP
            EXECUTE format(
                'UPDATE %I SET deleted_at = $1, modified_at = $1 WHERE espn_id = $2 AND deleted_at IS NULL',
                table_name
            )
            USING NEW.deleted_at, NEW.event_id;
        END LOOP;

        -- Soft delete records in tables that reference e_games.game_id (which references e_events.event_id)
        FOREACH table_name IN ARRAY tables_with_game_id LOOP
            EXECUTE format(
                'UPDATE %I SET deleted_at = $1, modified_at = $1 WHERE game_id = $2 AND deleted_at IS NULL',
                table_name
            )
            USING NEW.deleted_at, NEW.event_id;
        END LOOP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================================
-- 2. Table Definitions
-- ================================================

-- 2.1. Table: e_events
-- Stores event information from ESPN.
CREATE TABLE e_events (
    event_id INTEGER PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE,
    type INTEGER,
    short_name VARCHAR(12),
    home_team VARCHAR(7),
    away_team VARCHAR(7),
    home_team_normalized VARCHAR(7),
    away_team_normalized VARCHAR(7),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger: update modified_at on e_events before update
CREATE TRIGGER update_e_events_modified_at
BEFORE UPDATE ON e_events
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.2. Table: e_teams
-- Stores team information from ESPN.
CREATE TABLE e_teams (
    team_id INTEGER PRIMARY KEY,
    display_name VARCHAR NOT NULL,
    abbreviation VARCHAR NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on e_teams
CREATE TRIGGER trg_e_teams_modified_at
BEFORE UPDATE ON e_teams
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3. Table: e_players
-- Stores player information from ESPN.
CREATE TABLE e_players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on e_players
CREATE TRIGGER trg_e_players_modified_at
BEFORE UPDATE ON e_players
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.4. Table: e_games
-- Stores game information from ESPN.
CREATE TABLE e_games (
    game_id INTEGER PRIMARY KEY REFERENCES e_events(event_id),
    home_team_id INTEGER REFERENCES e_teams(team_id),
    away_team_id INTEGER REFERENCES e_teams(team_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on e_games
CREATE TRIGGER trg_e_games_modified_at
BEFORE UPDATE ON e_games
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.5. Table: e_game_players
-- Stores the relationship between games, players, and teams.
CREATE TABLE e_game_players (
    game_player_id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES e_games(game_id),
    player_id INTEGER REFERENCES e_players(player_id),
    team_id INTEGER REFERENCES e_teams(team_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT unique_game_players UNIQUE (game_id, player_id, team_id)
);

-- Trigger to update modified_at on e_game_players
CREATE TRIGGER trg_e_game_players_modified_at
BEFORE UPDATE ON e_game_players
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.6. Table: video_espn_mapping
-- Maps YouTube video IDs to ESPN event IDs.
CREATE TABLE video_espn_mapping (
    yt_id VARCHAR NOT NULL UNIQUE REFERENCES yt_metadata(video_id),
    espn_id INTEGER NOT NULL REFERENCES e_events(event_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (yt_id, espn_id),
    UNIQUE (yt_id, espn_id)
);

-- Trigger to update mapping
CREATE TRIGGER trg_manage_video_espn_mapping
AFTER INSERT OR UPDATE ON yt_video_file
FOR EACH ROW
EXECUTE FUNCTION manage_video_espn_mapping();

-- ================================================
-- 3. Soft Delete Triggers
-- ================================================

-- 3.1. Trigger to Invoke the Propagation Function on e_events
DROP TRIGGER IF EXISTS trg_propagate_soft_delete_e_events ON e_events;

CREATE TRIGGER trg_propagate_soft_delete_e_events
AFTER UPDATE ON e_events
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_e_events();

-- ================================================
-- 4. Index Definitions
-- ================================================

-- 4.1. Indexes for e_events


-- 4.2. Indexes for e_teams
CREATE INDEX idx_e_teams_display_name ON e_teams(display_name);
CREATE UNIQUE INDEX e_teams_abbreviation_unique ON e_teams(abbreviation);

-- 4.3. Indexes for e_players
CREATE INDEX idx_e_players_name ON e_players(name);

-- 4.4. Indexes for e_games


-- 4.5. Indexes for e_game_players
CREATE INDEX idx_e_game_players_game_id ON e_game_players(game_id);
CREATE INDEX idx_e_game_players_player_id ON e_game_players(player_id);
CREATE INDEX idx_e_game_players_team_id ON e_game_players(team_id);

-- 4.6 Indexes for video_espn_mapping
CREATE INDEX idx_yt_id ON video_espn_mapping (yt_id);
CREATE INDEX idx_espn_id ON video_espn_mapping (espn_id);
