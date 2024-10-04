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

-- 1.3 Trigger function to propagate soft deletes from exp_experiments to related tables
CREATE OR REPLACE FUNCTION propagate_soft_delete_experiments()
RETURNS TRIGGER AS $$
DECLARE
    tables_with_experiment_id TEXT[] := ARRAY[
        'exp_experiment_results',
        'exp_experiment_videos',
        'exp_experiment_test_cases',
        'exp_experiment_segments',
        'exp_experiment_prompt_terms',
        'exp_tests',
        'exp_test_prompt_terms'
    ];
    tables_with_test_id TEXT[] := ARRAY[
        'exp_test_transcriptions',
        'exp_test_transcription_segments',
        'exp_test_transcription_words'
    ];
    table_name TEXT;
    test_rec RECORD;
BEGIN
    IF NEW.deleted_at IS NOT NULL THEN
        -- Update tables that have experiment_id
        FOREACH table_name IN ARRAY tables_with_experiment_id LOOP
            EXECUTE format('UPDATE %I SET deleted_at = $1 WHERE experiment_id = $2 AND deleted_at IS NULL', table_name)
            USING NEW.deleted_at, NEW.experiment_id;
        END LOOP;

        -- Update related tables that have test_id
        FOR test_rec IN SELECT test_id FROM exp_tests WHERE experiment_id = NEW.experiment_id AND deleted_at IS NULL LOOP
            FOREACH table_name IN ARRAY tables_with_test_id LOOP
                EXECUTE format('UPDATE %I SET deleted_at = $1 WHERE test_id = $2 AND deleted_at IS NULL', table_name)
                USING NEW.deleted_at, test_rec.test_id;
            END LOOP;
        END LOOP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 1.4 Trigger function to propagate soft deletes from exp_test_prompts to related tables
CREATE OR REPLACE FUNCTION propagate_soft_delete_test_prompts()
RETURNS TRIGGER AS $$
DECLARE
    tables_with_test_id TEXT[] := ARRAY[
        'exp_test_transcriptions',
        'exp_test_transcription_segments',
        'exp_test_transcription_words',
        'exp_test_prompt_terms'
    ];
    table_name TEXT;
    test_rec RECORD;
BEGIN
    IF NEW.deleted_at IS NOT NULL THEN
        -- Update exp_tests where test_prompt_id matches
        EXECUTE 'UPDATE exp_tests SET deleted_at = $1 WHERE test_prompt_id = $2 AND deleted_at IS NULL'
        USING NEW.deleted_at, NEW.test_prompt_id;

        -- Update related tables that have test_id
        FOR test_rec IN SELECT test_id FROM exp_tests WHERE test_prompt_id = NEW.test_prompt_id AND deleted_at IS NULL LOOP
            FOREACH table_name IN ARRAY tables_with_test_id LOOP
                EXECUTE format('UPDATE %I SET deleted_at = $1 WHERE test_id = $2 AND deleted_at IS NULL', table_name)
                USING NEW.deleted_at, test_rec.test_id;
            END LOOP;
        END LOOP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 1.5 Trigger function to propagate soft deletes from exp_tests to related tables
CREATE OR REPLACE FUNCTION propagate_soft_delete_tests()
RETURNS TRIGGER AS $$
DECLARE
    tables_with_test_id TEXT[] := ARRAY[
        'exp_test_transcriptions',
        'exp_test_transcription_segments',
        'exp_test_transcription_words'
    ];
    table_name TEXT;
BEGIN
    IF NEW.deleted_at IS NOT NULL THEN
        -- Update tables that have test_id
        FOREACH table_name IN ARRAY tables_with_test_id LOOP
            EXECUTE format('UPDATE %I SET deleted_at = $1 WHERE test_id = $2 AND deleted_at IS NULL', table_name)
            USING NEW.deleted_at, NEW.test_id;
        END LOOP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ================================================
-- 2. Table Definitions
-- ================================================

-- 2.1. Create prompt_terms Table
CREATE TABLE IF NOT EXISTS prompt_terms (
    term VARCHAR(50) PRIMARY KEY,
    in_game_usage_score DECIMAL(3,2),
    general_speech_score DECIMAL(3,2),
    impact_transcription_score DECIMAL(3,2),
    confusion_potential_score DECIMAL(3,2),
    tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on prompt_terms
CREATE TRIGGER trg_prompt_terms_modified_at
BEFORE UPDATE ON prompt_terms
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.2. Create audio_segments Table
CREATE TABLE IF NOT EXISTS audio_segments (
    segment_id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL,
    segment_number INTEGER NOT NULL,
    start_time FLOAT,
    end_time FLOAT,
    snr DECIMAL(5,2),
    transcription_text_actual TEXT,
    raw_audio_path VARCHAR(500),
    processed_audio_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT unique_video_segments UNIQUE (video_id, segment_number)
);

-- Trigger to update modified_at on audio_segments
CREATE TRIGGER trg_audio_segments_modified_at
BEFORE UPDATE ON audio_segments
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3. EXPERIMENT SCHEMA

-- 2.3.1. Create exp_experiments Table
CREATE TABLE IF NOT EXISTS exp_experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on exp_experiments
CREATE TRIGGER trg_exp_experiments_modified_at
BEFORE UPDATE ON exp_experiments
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.2. Create exp_experiment_results Table
CREATE TABLE IF NOT EXISTS exp_experiment_results (
    experiment_id INTEGER PRIMARY KEY REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    total_tests INTEGER,
    average_wer DECIMAL(5,2),
    average_snr DECIMAL(5,2),
    average_probability DECIMAL(5,4),
    analysis_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on exp_experiment_results
CREATE TRIGGER trg_exp_experiment_results_modified_at
BEFORE UPDATE ON exp_experiment_results
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.3. Create exp_experiment_videos Table
CREATE TABLE IF NOT EXISTS exp_experiment_videos (
    experiment_id INTEGER NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    video_id VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (experiment_id, video_id)
);

-- Trigger to update modified_at on exp_experiment_videos
CREATE TRIGGER trg_exp_experiment_videos_modified_at
BEFORE UPDATE ON exp_experiment_videos
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.4. Create exp_experiment_test_cases Table
CREATE TABLE IF NOT EXISTS exp_experiment_test_cases (
    test_case_id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    test_case_name VARCHAR(255) NOT NULL,
    description TEXT,
    prompt_template TEXT,
    prompt_tokens INTEGER,
    is_dynamic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Trigger to update modified_at on exp_experiment_test_cases
CREATE TRIGGER trg_exp_experiment_test_cases_modified_at
BEFORE UPDATE ON exp_experiment_test_cases
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.5. Create exp_experiment_segments Table
CREATE TABLE IF NOT EXISTS exp_experiment_segments (
    experiment_id INTEGER NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    segment_id INTEGER NOT NULL REFERENCES audio_segments(segment_id) ON DELETE NO ACTION,
    test_case_id INTEGER NOT NULL REFERENCES exp_experiment_test_cases(test_case_id) ON DELETE NO ACTION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (experiment_id, segment_id)
);

-- Trigger to update modified_at on exp_experiment_segments
CREATE TRIGGER trg_exp_experiment_segments_modified_at
BEFORE UPDATE ON exp_experiment_segments
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.6. Create exp_experiment_prompt_terms Table
CREATE TABLE IF NOT EXISTS exp_experiment_prompt_terms (
    experiment_id INTEGER NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    term VARCHAR(50) NOT NULL REFERENCES prompt_terms(term) ON DELETE NO ACTION,
    in_game_usage_score_weighted DECIMAL(3,2),
    general_speech_score_weighted DECIMAL(3,2),
    impact_transcription_score_weighted DECIMAL(3,2),
    confusion_potential_score_weighted DECIMAL(3,2),
    final_score DECIMAL(3,2),
    tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (experiment_id, term)
);

-- Trigger to update modified_at on exp_experiment_prompt_terms
CREATE TRIGGER trg_exp_experiment_prompt_terms_modified_at
BEFORE UPDATE ON exp_experiment_prompt_terms
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.7. Create exp_test_prompts Table
CREATE TABLE IF NOT EXISTS exp_test_prompts (
    test_prompt_id SERIAL PRIMARY KEY,
    experiment_test_case_id INTEGER REFERENCES exp_experiment_test_cases(test_case_id),
    prompt TEXT,
    prompt_tokens INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Unique when prompt is NOT NULL and record is not soft-deleted
CREATE UNIQUE INDEX unique_prompt_tokens_not_null 
ON exp_test_prompts (prompt, prompt_tokens) 
WHERE prompt IS NOT NULL AND deleted_at IS NULL;

-- Unique when prompt is NULL and record is not soft-deleted
CREATE UNIQUE INDEX unique_prompt_tokens_null 
ON exp_test_prompts (prompt_tokens) 
WHERE prompt IS NULL AND deleted_at IS NULL;

-- Trigger to update modified_at on exp_test_prompts
CREATE TRIGGER trg_exp_test_prompts_modified_at
BEFORE UPDATE ON exp_test_prompts
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.8. Create exp_tests Table
CREATE TABLE exp_tests (
    test_id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES exp_experiments(experiment_id),
    segment_id INTEGER NOT NULL REFERENCES audio_segments(segment_id),
    test_case_id INTEGER NOT NULL REFERENCES exp_experiment_test_cases(test_case_id),
    test_prompt_id INTEGER REFERENCES exp_test_prompts(test_prompt_id),
    is_raw_audio boolean,
    average_avg_logprob NUMERIC(5,2),
    average_probability NUMERIC(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT unique_test_combination UNIQUE (experiment_id, segment_id, test_case_id, test_prompt_id, is_raw_audio)
);

-- Indexes for exp_tests
CREATE INDEX idx_exp_tests_experiment_id ON exp_tests(experiment_id int4_ops);
CREATE INDEX idx_exp_tests_segment_id ON exp_tests(segment_id int4_ops);
CREATE INDEX idx_exp_tests_test_case_id ON exp_tests(test_case_id int4_ops);
CREATE INDEX idx_exp_tests_test_prompt_id ON exp_tests(test_prompt_id int4_ops);

-- Trigger to update modified_at on exp_tests
CREATE TRIGGER trg_exp_tests_modified_at
  BEFORE UPDATE ON exp_tests
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();

-- 2.3.9. Create exp_test_prompt_terms Table
CREATE TABLE exp_test_prompt_terms (
    experiment_id INTEGER REFERENCES exp_experiments(experiment_id),
    term CHARACTER VARYING(50),
    test_id INTEGER REFERENCES exp_tests(test_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (experiment_id, term, test_id)
);

-- Indexes for exp_test_prompt_terms
CREATE INDEX idx_exp_test_prompt_terms_experiment_id ON exp_test_prompt_terms(experiment_id int4_ops);
CREATE INDEX idx_exp_test_prompt_terms_term ON exp_test_prompt_terms(term text_ops);
CREATE INDEX idx_exp_test_prompt_terms_test_id ON exp_test_prompt_terms(test_id int4_ops);
CREATE UNIQUE INDEX unique_exp_test_prompt_terms ON exp_test_prompt_terms(experiment_id int4_ops,term text_ops,test_id int4_ops);

-- Trigger to update modified_at on exp_test_prompt_terms
CREATE TRIGGER trg_exp_test_prompt_terms_modified_at
  BEFORE UPDATE ON exp_test_prompt_terms
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();

-- 2.3.10. Create exp_test_transcriptions Table
CREATE TABLE exp_test_transcriptions (
    transcription_id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES exp_tests(test_id),
    transcription_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for exp_test_transcriptions
CREATE UNIQUE INDEX idx_exp_test_transcriptions_test_id ON exp_test_transcriptions(test_id int4_ops);

-- Trigger to update modified_at on exp_test_transcriptions
CREATE TRIGGER trg_exp_test_transcriptions_modified_at
  BEFORE UPDATE ON exp_test_transcriptions
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();

-- 2.3.11. Create exp_test_transcription_segments Table
CREATE TABLE exp_test_transcription_segments (
    transcription_segment_id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES exp_tests(test_id),
    segment_number INTEGER,
    segment_text TEXT,
    start_time NUMERIC(10,4),
    end_time NUMERIC(10,4),
    avg_logprob NUMERIC(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for exp_test_transcription_segments
CREATE INDEX idx_exp_test_transcription_segments_test_id ON exp_test_transcription_segments(test_id int4_ops);
CREATE INDEX idx_exp_test_transcription_segments_segment_number ON exp_test_transcription_segments(segment_number int4_ops);
CREATE UNIQUE INDEX idx_exp_test_transcription_segments_test_id_segment_number ON exp_test_transcription_segments(test_id int4_ops, segment_number int4_ops);

-- Trigger to update modified_at on exp_test_transcription_segments
CREATE TRIGGER trg_exp_test_transcription_segments_modified_at
  BEFORE UPDATE ON exp_test_transcription_segments
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();

-- 2.3.12. Create exp_test_transcription_words Table
CREATE TABLE exp_test_transcription_words (
    transcription_word_id SERIAL PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES exp_tests(test_id),
    word_number INTEGER,
    word_text CHARACTER VARYING(255),
    start_time NUMERIC(10,4),
    end_time NUMERIC(10,4),
    probability NUMERIC(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for exp_test_transcription_words
CREATE INDEX idx_exp_test_transcription_words_test_id ON exp_test_transcription_words(test_id int4_ops);
CREATE INDEX idx_exp_test_transcription_words_word_number ON exp_test_transcription_words(word_number int4_ops);
CREATE UNIQUE INDEX idx_exp_test_transcription_words_test_id_word_number ON exp_test_transcription_words(test_id int4_ops, word_number int4_ops);

-- Trigger to update modified_at on exp_test_transcription_words
CREATE TRIGGER trg_exp_test_transcription_words_modified_at
  BEFORE UPDATE ON exp_test_transcription_words
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();

-- 2.4. VIDEO MAPPING SCHEMA

-- 2.4.1. Create video_espn_mapping Table
CREATE TABLE video_espn_mapping (
    yt_id VARCHAR NOT NULL UNIQUE REFERENCES yt_metadata(video_id),
    espn_id INTEGER NOT NULL REFERENCES e_events(event_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (yt_id, espn_id),
    UNIQUE (yt_id, espn_id)
);

-- Indexes for video_espn_mapping
CREATE INDEX idx_yt_id ON video_espn_mapping (yt_id);
CREATE INDEX idx_espn_id ON video_espn_mapping (espn_id);

-- Trigger to update mapping
CREATE TRIGGER trg_manage_video_espn_mapping
AFTER INSERT OR UPDATE ON yt_video_file
FOR EACH ROW EXECUTE FUNCTION manage_video_espn_mapping();


-- 2.5. ESPN SCHEMA

-- 2.5.1. Create e_teams table
CREATE TABLE e_teams (
    team_id INTEGER PRIMARY KEY,
    display_name VARCHAR NOT NULL,
    abbreviation VARCHAR NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for e_teams
CREATE INDEX idx_e_teams_display_name ON e_teams(display_name);
CREATE UNIQUE INDEX e_teams_abbreviation_unique ON e_teams(abbreviation);

-- Trigger to update modified_at on e_teams
CREATE TRIGGER trg_e_teams_modified_at
BEFORE UPDATE ON e_teams
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.5.2 Create e_players table
CREATE TABLE e_players (
    player_id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for e_players
CREATE INDEX idx_e_players_name ON e_players(name);

-- Trigger to update modified_at on e_players
CREATE TRIGGER trg_e_players_modified_at
BEFORE UPDATE ON e_players
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.5.3 Create e_games table
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

-- 2.5.4 Create e_game_players table
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

-- Indexes for e_game_players
CREATE INDEX idx_e_game_players_game_id ON e_game_players(game_id);
CREATE INDEX idx_e_game_players_player_id ON e_game_players(player_id);
CREATE INDEX idx_e_game_players_team_id ON e_game_players(team_id);

-- Trigger to update modified_at on e_game_players
CREATE TRIGGER trg_e_game_players_modified_at
BEFORE UPDATE ON e_game_players
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();


-- ================================================
-- 3. Soft Delete Triggers
-- ================================================

-- 3.1. Trigger to propagate soft deletes from exp_experiments to related tables
DROP TRIGGER IF EXISTS trg_exp_experiments_soft_delete ON exp_experiments;

CREATE TRIGGER trg_exp_experiments_soft_delete
AFTER UPDATE ON exp_experiments
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_experiments();

-- 3.2. Trigger to propagate soft deletes from exp_test_prompts to related tables
DROP TRIGGER IF EXISTS trg_exp_test_prompts_soft_delete ON exp_test_prompts;

CREATE TRIGGER trg_exp_test_prompts_soft_delete
AFTER UPDATE ON exp_test_prompts
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_test_prompts();

-- 3.3. Trigger to propagate soft deletes from exp_tests to related tables
DROP TRIGGER IF EXISTS trg_exp_tests_soft_delete ON exp_tests;

CREATE TRIGGER trg_exp_tests_soft_delete
AFTER UPDATE ON exp_tests
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_tests();

-- ================================================
-- 4. Index Definitions
-- ================================================

-- 4.1. Indexes for exp_experiment_videos
CREATE INDEX IF NOT EXISTS idx_exp_experiment_videos_experiment_id 
ON exp_experiment_videos(experiment_id);

CREATE INDEX IF NOT EXISTS idx_exp_experiment_videos_video_id 
ON exp_experiment_videos(video_id);

-- 4.2. Indexes for exp_experiment_segments
CREATE INDEX IF NOT EXISTS idx_exp_experiment_segments_experiment_id 
ON exp_experiment_segments(experiment_id);

CREATE INDEX IF NOT EXISTS idx_exp_experiment_segments_segment_id 
ON exp_experiment_segments(segment_id);

CREATE INDEX IF NOT EXISTS idx_exp_experiment_segments_test_case_id 
ON exp_experiment_segments(test_case_id);

-- 4.3. Indexes for exp_experiment_prompt_terms
CREATE INDEX IF NOT EXISTS idx_exp_experiment_prompt_terms_experiment_id 
ON exp_experiment_prompt_terms(experiment_id);

CREATE INDEX IF NOT EXISTS idx_exp_experiment_prompt_terms_term 
ON exp_experiment_prompt_terms(term);

-- 4.4. Indexes for exp_tests
CREATE INDEX IF NOT EXISTS idx_exp_tests_experiment_id 
ON exp_tests(experiment_id);

CREATE INDEX IF NOT EXISTS idx_exp_tests_segment_id 
ON exp_tests(segment_id);

CREATE INDEX IF NOT EXISTS idx_exp_tests_test_case_id 
ON exp_tests(test_case_id);

CREATE INDEX IF NOT EXISTS idx_exp_tests_test_prompt_id 
ON exp_tests(test_prompt_id);

-- 4.5. Indexes for exp_test_prompt_terms
CREATE INDEX IF NOT EXISTS idx_exp_test_prompt_terms_experiment_id 
ON exp_test_prompt_terms(experiment_id);

CREATE INDEX IF NOT EXISTS idx_exp_test_prompt_terms_term 
ON exp_test_prompt_terms(term);

CREATE INDEX IF NOT EXISTS idx_exp_test_prompt_terms_test_id 
ON exp_test_prompt_terms(test_id);

-- 4.6. Indexes for exp_test_transcriptions
CREATE INDEX IF NOT EXISTS idx_exp_test_transcriptions_test_id 
ON exp_test_transcriptions(test_id);

-- 4.7. Indexes for exp_test_transcription_segments
CREATE INDEX IF NOT EXISTS idx_exp_test_transcription_segments_test_id 
ON exp_test_transcription_segments(test_id);

CREATE INDEX IF NOT EXISTS idx_exp_test_transcription_segments_segment_number 
ON exp_test_transcription_segments(segment_number);

-- 4.8. Indexes for exp_test_transcription_words
CREATE INDEX IF NOT EXISTS idx_exp_test_transcription_words_test_id 
ON exp_test_transcription_words(test_id);

CREATE INDEX IF NOT EXISTS idx_exp_test_transcription_words_word_number 
ON exp_test_transcription_words(word_number);

-- 4.9. Indexes for prompt_terms
CREATE INDEX IF NOT EXISTS idx_prompt_terms_term 
ON prompt_terms(term);

-- 4.10. Indexes for audio_segments
CREATE INDEX IF NOT EXISTS idx_audio_segments_video_id 
ON audio_segments(video_id);

-- 4.11. Indexes for exp_experiments
CREATE INDEX IF NOT EXISTS idx_exp_experiments_experiment_name 
ON exp_experiments(experiment_name);

-- 4.12. Indexes for exp_test_prompts
CREATE INDEX IF NOT EXISTS idx_exp_test_prompts_prompt
ON exp_test_prompts(prompt);

CREATE INDEX IF NOT EXISTS idx_exp_test_prompts_prompt_tokens
ON exp_test_prompts(prompt_tokens);
