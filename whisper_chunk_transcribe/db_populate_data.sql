-- Description: This SQL script populates the database with sample data for running transcription experiments.
-- It creates a new experiment, selects a random 1% of videos, assigns test cases to the experiment segments, 
-- and calculates prompt scores based on weighted factors. Finally, it checks for distribution bias across the test cases.

---------------------------------
-- Statement 1: 
-- Creates a new experiment, randomly selects 1% of videos, links audio segments with the experiment,
-- and calculates weighted prompt scores based on specified categories.
---------------------------------

WITH date_range AS (
    -- Define the date range for selecting videos
    SELECT
        '2014-01-01 00:00:00'::timestamp AS start_date,
        '2018-12-31 23:59:59'::timestamp AS end_date
),
-- Insert a new experiment record and retrieve its experiment_id
inserted_experiment AS (
    INSERT INTO exp_experiments (experiment_name, description)
    VALUES (
        'Prompt Impact Experiment',
        'An experiment to evaluate the impact of different prompts on audio transcription accuracy.'
    )
    RETURNING experiment_id
),
total_rows AS (
    -- Calculate the total number of rows within the selected date range
    SELECT COUNT(*) AS cnt
    FROM yt_metadata AS m
    JOIN yt_video_file AS file ON m.video_id = file.video_id
    WHERE m.event_date_local_time BETWEEN (SELECT start_date FROM date_range) AND (SELECT end_date FROM date_range)
),
-- Insert random 1% of videos into exp_experiment_videos, linking them to the new experiment
inserted_videos AS (
    INSERT INTO exp_experiment_videos (experiment_id, video_id)
    SELECT
        (SELECT experiment_id FROM inserted_experiment),
        m.video_id
    FROM yt_metadata AS m
    JOIN yt_video_file AS file ON m.video_id = file.video_id
    WHERE m.event_date_local_time BETWEEN (SELECT start_date FROM date_range) AND (SELECT end_date FROM date_range)
    ORDER BY RANDOM()
    LIMIT (SELECT CEIL(cnt * 0.01) FROM total_rows) -- Select 1% of the total videos
    RETURNING video_id
),
-- Link audio segments to the experiment in exp_experiment_segments
inserted_segments AS (
    INSERT INTO exp_experiment_segments (experiment_id, segment_id)
    SELECT
        (SELECT experiment_id FROM inserted_experiment),
        seg.segment_id
    FROM audio_segments seg
    JOIN inserted_videos iv ON seg.video_id = iv.video_id
    RETURNING experiment_id, segment_id
)
-- Define weights for the prompt scoring categories (adjustable)
prompt_category_weights AS (
    SELECT 
        1.0 AS in_game_usage,       -- Weight for in-game usage relevance
        1.0 AS general_speech,      -- Weight for general speech applicability
        1.0 AS impact_transcription, -- Weight for impact on transcription accuracy
        1.0 AS confusion_potential  -- Weight for confusion potential
),
-- Calculate final scores based on prompt terms and weights
scoring AS (
    SELECT 
        pt.term,
        pt.tokens,

        -- Base scores
        -- pt.in_game_usage_score,
        -- pt.general_speech_score,
        -- pt.impact_transcription_score,
        -- pt.confusion_potential_score,
        
        -- Weighted scores for each category
        (pt.in_game_usage_score * w.in_game_usage) AS in_game_usage_result,
        (pt.general_speech_score * w.general_speech) AS general_speech_result,
        (pt.impact_transcription_score * w.impact_transcription) AS impact_transcription_result,
        (pt.confusion_potential_score * w.confusion_potential) AS confusion_potential_result,
        
        -- Final score calculation
        (
            (pt.in_game_usage_score * w.in_game_usage) + 
            (pt.general_speech_score * w.general_speech)
        ) * (
            (pt.impact_transcription_score * w.impact_transcription) + 
            (pt.confusion_potential_score * w.confusion_potential)
        ) AS final_score

    FROM prompt_terms pt, prompt_category_weights w
),
exp_prompt_terms AS (
    -- Insert calculated prompt scores into exp_prompt_terms
    INSERT INTO exp_prompt_terms (
        term, 
        experiment_id, 
        in_game_usage_score_weighted, 
        general_speech_score_weighted, 
        impact_transcription_score_weighted, 
        confusion_potential_score_weighted, 
        final_score, 
        tokens
    )
    SELECT 
        term, 
        (SELECT experiment_id FROM inserted_experiment),
        in_game_usage_result, 
        general_speech_result, 
        impact_transcription_result, 
        confusion_potential_result, 
        final_score, 
        tokens
    FROM scoring
);


---------------------------------
-- Statement 2: 
-- Inserts test cases for the experiment, including different prompt templates for testing.
---------------------------------

WITH inserted_experiment AS (
    SELECT 2 AS experiment_id -- Assume experiment_id for simplicity in this step
),
-- Insert test cases for various transcription scenarios
inserted_test_cases AS (
    INSERT INTO exp_experiment_test_cases (
        experiment_id,
        test_case_name,
        description,
        prompt_template,
        prompt_tokens,
        is_dynamic,
        use_prev_transcription
        )
    VALUES
        (
            (SELECT experiment_id FROM inserted_experiment),
            'No Prompt',
            'Transcribe the audio without any additional prompt.',
            NULL,
            0,
            FALSE,  -- is_dynamic
            FALSE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Unrelated Prompt',
            'Use an unrelated prompt to assess its impact on transcription accuracy.', 
            'A comprehensive guide to planting and maintaining a thriving home garden, tailored for both beginners and experienced gardeners. This audio will focus on essential gardening terms and practices that contribute to a successful and sustainable garden. Key terms include: soil preparation, the importance of testing soil pH, selecting quality seeds based on climate and season, understanding proper watering techniques to avoid overwatering or underwatering, ensuring adequate sunlight for different types of plants, effective pruning methods to encourage healthy growth, understanding various types of fertilizers and their application, the benefits of composting for soil enrichment, promoting healthy plant growth through nutrient management, harvesting crops at the right time for maximum flavor and yield, implementing integrated pest control strategies to protect plants from damage, monitoring plant health for early detection of diseases, the importance of mulch for moisture retention and weed control, principles of organic gardening that focus on sustainability, and cultivating a diverse array of plants such as vegetables, fruits, herbs, and flowers to create a vibrant and productive garden ecosystem. Mastering these concepts will not only enhance your gardening skills but also contribute to a more sustainable environment. Whether you are growing food for your family or beautifying your outdoor space, these practices will lead you toward gardening success.',

            247, -- Token length calculation (use `python -m whisper_chunk_transcribe.helper_determine_prompt_length_tokens "{prompt_template}"` helper script)
            FALSE,  -- is_dynamic
            FALSE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Generic Sports Prompt',
            'Provide a general sports-related prompt without specific sports terminology.',
            'An overview of the fundamental concepts and strategies involved in various sports, emphasizing the importance of teamwork, discipline, and physical fitness. This guide will cover essential themes such as the significance of practice and preparation, understanding rules and regulations, developing skills and techniques, and the role of coaching and mentorship in athlete development. Key components include maintaining physical conditioning, fostering a positive mindset, enhancing communication among team members, and setting personal and collective goals. Additionally, the prompt will highlight the value of sportsmanship, respect for opponents, and the ability to handle both victories and defeats gracefully. Engaging in sports promotes not only physical health but also mental resilience and social interaction. It encourages individuals to push their limits, develop a sense of commitment, and build lasting relationships through shared experiences. Whether participating in recreational activities or competitive events, embracing these principles contributes to a fulfilling sports experience. By understanding the broader implications of sports in our lives, individuals can cultivate a lifelong appreciation for physical activity, personal growth, and the joy of movement, while also recognizing the impact of sports on community building and cultural exchange. This holistic approach will inspire athletes of all levels to strive for excellence and personal growth.',
            252, -- Token length calculation (use `python -m whisper_chunk_transcribe.helper_determine_prompt_length_tokens "{prompt_template}"` helper script)
            FALSE,  -- is_dynamic
            FALSE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Baseball Terms',
            'Include common baseball terms in the prompt to provide relevant context.',
            -- Include the result from the `python -m whisper_chunk_transcribe.helper_update_prompt_terms` helper script
            'terms: pitcher, home run, inning, catcher, innings, fly ball, bases loaded, strikeout, batter, fastball, RBI, foul ball, second baseman, first baseman, shortstop, third baseman, left fielder, line drive, right fielder, center fielder, pitch count, double play, dugout, outfield, infield, at bat, grounder, pop fly, bullpen, umpire, changeup, pitch, relief pitcher, curveball, strike, bunt, sacrifice fly, designated hitter, slider, on deck, intentional walk, ERA, walk-off, foul tip, tag out, full count, no-hitter, pinch hitter, OBP, slugging percentage, pick-off, force out, sinker, pinch runner, perfect game, grand slam, fly out, balk, go-ahead run, mound visit, earned run, knuckleball, fielder''s choice, warning track, 1-2, 2-1, 2-2, 2-0, 3-1, 3-0, 3-2, 0-1, 0-2, 1-0, 1-1, run-down, shutout, caught looking, long ball, inside pitch, complete game'
            252, -- Token length calculation (use `python -m whisper_chunk_transcribe.helper_determine_prompt_length_tokens "{prompt_template}"` helper script)
            FALSE,  -- is_dynamic
            FALSE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Baseball Prompt with Game-Specific Details',
            'Use a prompt that includes team names and player names.',
            'The {teamA} are playing the {teamB}. The players are {playerA} and {playerB}.',
            NULL,
            TRUE,  -- is_dynamic
            FALSE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Previous Segment Transcription',
            'Use the transcription of the previous audio segment as the prompt.',
            '{previous_transcription}',
            NULL,
            TRUE,  -- is_dynamic
            TRUE -- use_prev_transcription
        ),(
            (SELECT experiment_id FROM inserted_experiment),
            'Combined Prompt with Previous Transcription and Game Details',
            'Combine the previous segment''s transcription with baseball-specific terms and game-specific details.',
            'The {teamA} are playing the {teamB}. The players are {playerA} and {playerB}. {previous_transcription}.',
            NULL,
            TRUE,  -- is_dynamic
            TRUE -- use_prev_transcription
        )
),
-- Prepopulate exp_test_prompts for test cases that do not use previous transcriptions
inserted_prompt_terms AS (
    INSERT INTO exp_test_prompts (
        experiment_test_case_id,
        prompt,
        prompt_tokens
    )
    SELECT
        etc.test_case_id,
        etc.prompt_template,
        etc.prompt_tokens
    FROM exp_experiment_test_cases etc
    WHERE etc.use_prev_transcription = FALSE
        AND etc.experiment_id = (SELECT experiment_id FROM inserted_experiment)
    ORDER BY etc.test_case_id
);


---------------------------------
-- Statement 3: 
-- Randomly assigns test cases to experiment segments to form treatment groups.
---------------------------------

-- Update exp_experiment_segments with random test_case_ids
UPDATE exp_experiment_segments es
SET test_case_id = subquery.test_case_id
FROM (
    SELECT 
        es.experiment_id,
        es.segment_id,
        tc.test_case_id
    FROM exp_experiment_segments es
    JOIN exp_test_cases tc ON tc.experiment_id = es.experiment_id
    ORDER BY RANDOM() -- Assign test cases randomly
) AS subquery
WHERE es.experiment_id = subquery.experiment_id AND es.segment_id = subquery.segment_id;


---------------------------------
-- Statement 4: 
-- Evaluate the distribution of segments across test cases to detect biases.
---------------------------------
-- This step helps ensure that test cases are evenly distributed across segments, reducing bias.

WITH test_case_counts AS (
    -- Count the number of segments per test case
    SELECT 
        test_case_id, 
        COUNT(*) AS count
    FROM exp_experiment_segments
    GROUP BY test_case_id
),
total_count AS (
    -- Get the total segment count
    SELECT SUM(count) AS total
    FROM test_case_counts
),
expected_count AS (
    -- Calculate the percentage distribution and bias
    SELECT 
        test_case_id,
        count,
        ROUND(count * 100.0 / total::numeric, 2) AS pct_of_total,
        (count - (total / (SELECT COUNT(*) FROM test_case_counts))) AS bias
    FROM test_case_counts, total_count
)
-- Output the distribution and bias analysis
SELECT 
    test_case_id, 
    count AS cnt_segments,
    pct_of_total,
    ROUND(bias, 0) AS cnt_away_from_perf_dist, -- Bias from perfect distribution. When showing more decimal places the sum will approach 0.0
    CASE 
        WHEN ROUND(bias, 0) > 0 THEN 'Over Represented'
        WHEN ROUND(bias, 0) < 0 THEN 'Under Represented'
        ELSE 'Perfectly Distributed'
    END AS distribution_bias
FROM expected_count
ORDER BY test_case_id;
