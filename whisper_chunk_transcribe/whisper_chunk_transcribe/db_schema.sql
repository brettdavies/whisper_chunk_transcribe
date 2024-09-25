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
    tokens INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
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
    segment_number INT NOT NULL,
    start_time FLOAT,
    end_time FLOAT,
    snr DECIMAL(5,2),
    transcription_text_actual TEXT,
    raw_audio_path VARCHAR(500),
    processed_audio_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
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
    deleted_at TIMESTAMP
);

-- Trigger to update modified_at on exp_experiments
CREATE TRIGGER trg_exp_experiments_modified_at
BEFORE UPDATE ON exp_experiments
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.2. Create exp_experiment_results Table
CREATE TABLE IF NOT EXISTS exp_experiment_results (
    experiment_id INT PRIMARY KEY REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    total_tests INT,
    average_wer DECIMAL(5,2),
    average_snr DECIMAL(5,2),
    average_probability DECIMAL(5,4),
    analysis_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Trigger to update modified_at on exp_experiment_results
CREATE TRIGGER trg_exp_experiment_results_modified_at
BEFORE UPDATE ON exp_experiment_results
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.3. Create exp_experiment_videos Table
CREATE TABLE IF NOT EXISTS exp_experiment_videos (
    experiment_id INT NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    video_id VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
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
    experiment_id INT NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    test_case_name VARCHAR(255) NOT NULL,
    description TEXT,
    prompt_template TEXT,
    prompt_tokens INT,
    is_dynamic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Trigger to update modified_at on exp_experiment_test_cases
CREATE TRIGGER trg_exp_experiment_test_cases_modified_at
BEFORE UPDATE ON exp_experiment_test_cases
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.5. Create exp_experiment_segments Table
CREATE TABLE IF NOT EXISTS exp_experiment_segments (
    experiment_id INT NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    segment_id INT NOT NULL REFERENCES audio_segments(segment_id) ON DELETE NO ACTION,
    test_case_id INT NOT NULL REFERENCES exp_experiment_test_cases(test_case_id) ON DELETE NO ACTION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    PRIMARY KEY (experiment_id, segment_id)
);

-- Trigger to update modified_at on exp_experiment_segments
CREATE TRIGGER trg_exp_experiment_segments_modified_at
BEFORE UPDATE ON exp_experiment_segments
FOR EACH ROW
EXECUTE FUNCTION update_modified_at_column();

-- 2.3.6. Create exp_experiment_prompt_terms Table
CREATE TABLE IF NOT EXISTS exp_experiment_prompt_terms (
    experiment_id INT NOT NULL REFERENCES exp_experiments(experiment_id) ON DELETE NO ACTION,
    term VARCHAR(50) NOT NULL,
    in_game_usage_score_weighted DECIMAL(3,2),
    general_speech_score_weighted DECIMAL(3,2),
    impact_transcription_score_weighted DECIMAL(3,2),
    confusion_potential_score_weighted DECIMAL(3,2),
    final_score DECIMAL(3,2),
    tokens INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
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
    prompt_tokens INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
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
CREATE UNIQUE INDEX exp_tests_pkey ON exp_tests(test_id int4_ops);

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
    CONSTRAINT exp_test_prompt_terms_pkey PRIMARY KEY (experiment_id, term, test_id)
);

-- Indexes for exp_test_prompt_terms
CREATE INDEX idx_exp_test_prompt_terms_experiment_id ON exp_test_prompt_terms(experiment_id int4_ops);
CREATE INDEX idx_exp_test_prompt_terms_term ON exp_test_prompt_terms(term text_ops);
CREATE INDEX idx_exp_test_prompt_terms_test_id ON exp_test_prompt_terms(test_id int4_ops);
CREATE UNIQUE INDEX exp_test_prompt_terms_pkey ON exp_test_prompt_terms(experiment_id int4_ops,term text_ops,test_id int4_ops);
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
CREATE UNIQUE INDEX exp_test_transcriptions_pkey ON exp_test_transcriptions(transcription_id int4_ops);

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
CREATE UNIQUE INDEX exp_test_transcription_segments_pkey ON exp_test_transcription_segments(transcription_segment_id int4_ops);

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
CREATE UNIQUE INDEX exp_test_transcription_words_pkey ON exp_test_transcription_words(transcription_word_id int4_ops);

-- Trigger to update modified_at on exp_test_transcription_words
CREATE TRIGGER trg_exp_test_transcription_words_modified_at
  BEFORE UPDATE ON exp_test_transcription_words
  FOR EACH ROW
  EXECUTE FUNCTION update_modified_at_column();




-- ================================================
-- 3. Soft Delete Triggers
-- ================================================

-- 3.1. Trigger to propagate soft deletes from exp_experiments to related tables
CREATE TRIGGER trg_exp_experiments_soft_delete
AFTER UPDATE ON exp_experiments
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_to_related_tables_exp();

-- 3.2. Trigger to propagate soft deletes from exp_test_prompts to related tables
CREATE TRIGGER trg_exp_test_prompts_soft_delete
AFTER UPDATE ON exp_test_prompts
FOR EACH ROW
WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
EXECUTE FUNCTION propagate_soft_delete_to_related_tables_exp();

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
