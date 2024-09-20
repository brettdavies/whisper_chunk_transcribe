-- Create the Initial Prompt Term Scores table
CREATE TABLE IF NOT EXISTS prompt_terms (
    term VARCHAR(50) PRIMARY KEY,
    in_game_usage_score DECIMAL(3,2),
    general_speech_score DECIMAL(3,2),
    impact_transcription_score DECIMAL(3,2),
    confusion_potential_score DECIMAL(3,2),
    tokens INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_segments (
    segment_id SERIAL PRIMARY KEY,
    video_id character varying(255),
    segment_number INT,
    start_time FLOAT,
    end_time FLOAT,
    snr DECIMAL(5,2),
    transcription_text_actual TEXT,
    raw_audio_path VARCHAR(500),
    processed_audio_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_video_segments UNIQUE (video_id, segment_number)  -- Unique constraint
);

-- EXPERIMENT SCHEMA
CREATE TABLE IF NOT EXISTS exp_experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_experiment_videos (
    experiment_id INT REFERENCES exp_experiments(experiment_id),
    video_id VARCHAR(500),
    PRIMARY KEY (experiment_id, video_id)
);

CREATE TABLE IF NOT EXISTS exp_test_cases (
    test_case_id SERIAL PRIMARY KEY,
    test_case_name VARCHAR(255),
    description TEXT,
    prompt_template TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_tests (
    test_id SERIAL PRIMARY KEY,
    experiment_id INT REFERENCES exp_experiments(experiment_id),
    segment_id INT REFERENCES audio_segments(segment_id),
    test_case_id INT REFERENCES exp_test_cases(test_case_id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    used_raw_audio BOOLEAN,
    prompt TEXT,
    prompt_tokens INT,
    wer DECIMAL(5,2),
    snr DECIMAL(5,2),
    complement_avg_logprob DECIMAL(5,2),
    average_probability DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_transcriptions (
    transcription_id SERIAL PRIMARY KEY,
    test_id INT REFERENCES exp_tests(test_id),
    transcription_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_transcription_segments (
    segment_id SERIAL PRIMARY KEY,
    transcription_id INT REFERENCES exp_transcriptions(transcription_id),
    segment_number INT,
    segment_text TEXT,
    start_time DECIMAL(10,4),
    end_time DECIMAL(10,4),
    avg_logprob DECIMAL(5,2),
    complement_avg_logprob DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_transcription_words (
    word_id SERIAL PRIMARY KEY,
    segment_id INT REFERENCES exp_transcription_segments(segment_id),
    word_number INT,
    word_text VARCHAR(255),
    start_time DECIMAL(10,4),
    end_time DECIMAL(10,4),
    probability DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_prompt_terms (
    term VARCHAR(50),
    experiment_id INT REFERENCES exp_experiments(experiment_id),
    in_game_usage_score DECIMAL(3,2),
    general_speech_score DECIMAL(3,2),
    impact_transcription_score DECIMAL(3,2),
    confusion_potential_score DECIMAL(3,2),
    final_score DECIMAL(3,2),
    tokens INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (experiment_id, term)
);

CREATE TABLE IF NOT EXISTS exp_test_prompt_terms (
    test_id INT REFERENCES exp_tests(test_id),
    term VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (test_id, term)
);

CREATE TABLE IF NOT EXISTS exp_experiment_results (
    experiment_id INT PRIMARY KEY REFERENCES exp_experiments(experiment_id),
    total_tests INT,
    average_wer DECIMAL(5,2),
    average_snr DECIMAL(5,2),
    average_complement_avg_logprob DECIMAL(5,2),
    average_probability DECIMAL(5,4),
    analysis_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- -- Create indexes
-- CREATE INDEX idx_exp_prompt_term_scores_term ON exp_prompt_term_scores(term);
-- CREATE INDEX idx_exp_prompt_term_category_weights_experiment_id ON exp_prompt_term_category_weights(experiment_id);
-- CREATE INDEX idx_exp_prompt_term_tokens_term ON exp_prompt_term_tokens(term);
-- CREATE INDEX idx_exp_prompt_term_tokens_experiment_id ON exp_prompt_term_tokens(experiment_id);
-- CREATE INDEX idx_exp_prompt_term_tokens_experiment_id_term ON exp_prompt_term_tokens(experiment_id, term);

-- -- Create custom constraint indexes
-- CREATE UNIQUE INDEX exp_prompt_term_tokens_experiment_id_term_key ON exp_prompt_term_tokens(experiment_id, term);  -- Ensuring uniqueness for (experiment_id, term)

-- -- Trigger function to update modified_at column
-- CREATE OR REPLACE FUNCTION update_modified_at_column_exp()
-- RETURNS TRIGGER AS $$
-- BEGIN
--     NEW.modified_at = CURRENT_TIMESTAMP;
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- -- Trigger to call the update modified_at function on update
-- CREATE TRIGGER update_exp_prompt_term_scores_modified_at
-- BEFORE UPDATE ON exp_prompt_term_scores
-- FOR EACH ROW
-- EXECUTE FUNCTION update_modified_at_column_exp();

-- CREATE TRIGGER update_exp_prompt_term_category_weights_modified_at
-- BEFORE UPDATE ON exp_prompt_term_category_weights
-- FOR EACH ROW
-- EXECUTE FUNCTION update_modified_at_column_exp();

-- CREATE TRIGGER update_exp_prompt_term_tokens_modified_at
-- BEFORE UPDATE ON exp_prompt_term_tokens
-- FOR EACH ROW
-- EXECUTE FUNCTION update_modified_at_column_exp();

-- -- Trigger function to propagate soft delete to related tables
-- CREATE OR REPLACE FUNCTION propagate_soft_delete_to_related_tables_exp()
-- RETURNS TRIGGER AS $$
-- DECLARE
--     tables TEXT[] := ARRAY[''exp_prompt_term_tokens'];
--     table_name TEXT;
-- BEGIN
--     IF NEW.deleted_at IS NOT NULL THEN
--         FOREACH table_name IN ARRAY tables LOOP
--             EXECUTE format('UPDATE %I SET deleted_at = $1 WHERE experiment_id = $2', table_name)
--             USING NEW.deleted_at, NEW.experiment_id;
--         END LOOP;
--     END IF;
--     RETURN NEW;
-- END;
-- $$ LANGUAGE plpgsql;

-- -- Trigger to call the soft delete function on update
-- CREATE TRIGGER exp_prompt_term_category_weights_soft_delete
-- AFTER UPDATE ON exp_prompt_term_category_weights
-- FOR EACH ROW
-- WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
-- EXECUTE FUNCTION propagate_soft_delete_to_related_tables_exp();

-- CREATE TRIGGER propagate_exp_prompt_term_category_weights_soft_delete
-- AFTER UPDATE ON exp_prompt_term_category_weights
-- FOR EACH ROW
-- WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
-- EXECUTE FUNCTION propagate_soft_delete_to_related_tables_exp();

-- -- Trigger to call the delete function on insert
-- CREATE TRIGGER after_exp_prompt_term_scores_insert
-- AFTER INSERT ON exp_prompt_term_scores
-- FOR EACH ROW
-- EXECUTE FUNCTION delete_from_experiment_ids();
