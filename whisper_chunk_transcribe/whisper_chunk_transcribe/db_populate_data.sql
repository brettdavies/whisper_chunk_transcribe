-- Description: This script populates the database with sample data for testing purposes.
-- It inserts test casees, creates a new experiment, and creates a random 1% treatment group for experiments.

-- Create test cases
INSERT INTO exp_test_cases (
    test_case_id,
    test_case_name,
    description,
    prompt_template
) VALUES
(
    1,
    'No Prompt',
    'Transcribe the audio without any additional prompt.',
    NULL
),
(
    2,
    'Unrelated Prompt',
    'Use an unrelated prompt to assess its impact on transcription accuracy.',
    'A guide to planting and maintaining a home garden. Focus on key terms: soil, seeds, watering, sunlight, pruning, fertilizer, compost, growth, harvest, pest control, plant health, mulch, organic gardening, vegetables, fruits, herbs, and flowers.'
),
(
    3,
    'Generic Sports Prompt',
    'Provide a general sports-related prompt without specific sports terminology.',
    'The athletes are displaying exceptional skill and teamwork in today''s match. The competition is intense, and the outcome is highly anticipated by fans around the world.'
),
(
    4,
    'Generic Sports Prompt with Baseball Terms',
    'Include common baseball terms in the prompt to provide relevant context.',
    'Key terms: {key_terms}'
),
(
    5,
    'Baseball Prompt with Game-Specific Details',
    'Use a prompt that includes baseball terms and specific details about the game.',
    'The game between {team_a} and {team_b} is underway at {stadium_name}. Star players like {player_a} and {player_b} are expected to make significant impacts. Key terms: {key_terms}'
),
(
    6,
    'Previous Segment Transcription',
    'Use the transcription of the previous audio segment as the prompt.',
    '{previous_transcription}'
),
(
    7,
    'Combined Prompt with Previous Transcription and Game Details',
    'Combine the previous segment''s transcription with baseball-specific terms and game-specific details.',
    '{previous_transcription} Continuing the matchup between {team_a} and {team_b} at {stadium_name}. Notable players like {player_a} are showing impressive performances. Key terms: {key_terms}'
);

-- Create experiment record and get the experiment_id
WITH inserted_experiment AS (
    INSERT INTO exp_experiments (experiment_name, description)
    VALUES (
        'Prompt Impact Experiment',
        'An experiment to evaluate the impact of different prompts on audio transcription accuracy.'
    )
    RETURNING experiment_id
),
total_rows AS (
    SELECT COUNT(*) AS cnt
    FROM yt_metadata AS m
    JOIN yt_video_file AS file ON m.video_id = file.video_id
    WHERE m.event_date_local_time BETWEEN '2014-01-01 00:00:00' AND '2018-12-31 23:59:59'
)
INSERT INTO exp_experiment_videos (experiment_id, video_id)
SELECT
    (SELECT experiment_id FROM inserted_experiment),
    m.video_id
FROM yt_metadata AS m
JOIN yt_video_file AS file ON m.video_id = file.video_id
WHERE m.event_date_local_time BETWEEN '2014-01-01 00:00:00' AND '2018-12-31 23:59:59'
ORDER BY RANDOM()
LIMIT (SELECT CEIL(cnt * 0.01) FROM total_rows);