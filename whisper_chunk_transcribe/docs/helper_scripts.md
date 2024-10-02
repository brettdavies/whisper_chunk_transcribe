# Helper Script Documentation

This document provides a simple guide to three helper scripts used in the transcription and experiment workflow. Each script performs specific tasks, and the document explains what the script does, how to run it, and when it should be called in the overall workflow.

---

## 1. `helper_determine_prompt_length_tokens.py`

### Link
[helper_determine_prompt_length_tokens.py](../whisper_chunk_transcribe/helper_determine_prompt_length_tokens.py)

### Description:
This script determines the token length of a given prompt using the Whisper transcription model. It helps ensure that prompt lengths do not exceed the token limits for transcription experiments.

### How to Call:
To run the script with a specific prompt:
```bash
python -m whisper_chunk_transcribe.helper_determine_prompt_length_tokens.py "Your prompt text here"
```

### When to Use:
- **Before setting up experiments** that require token length calculations for prompt templates.
- **During experiment design** to ensure that prompts fit within the token limit of the transcription engine.

### Summary of Workflow:
- Uses the Whisper model to calculate the token length of a provided prompt.
- Outputs the token length to ensure it does not exceed the allowed limit.

---

## 2. `helper_update_prompt_terms.py`

### Link
[helper_update_prompt_terms.py](../whisper_chunk_transcribe/helper_update_prompt_terms.py)

### Description:
This script updates the prompt terms for an experiment by retrieving terms from the database, calculating their token lengths, and building a prompt that fits within a specified token limit.

### How to Call:
To run the script with an experiment ID and optional token limit:
```bash
python -m whisper_chunk_transcribe.helper_update_prompt_terms --experiment_id 1 --max_tokens 255
```

### When to Use:
- **After populating the experiment data** but before running the experiment, to build the final prompts for any relevant test cases.
- **During prompt creation**, to ensure that the selected terms fit within the token limit for the transcription engine.

### Summary of Workflow:
- Retrieves prompt terms from the database for the specified `experiment_id`.
- Builds a prompt by adding terms that do not exceed the `max_tokens` limit.
- Outputs the final prompt and its token count.

---

## 3. `helper_player_team_names.py`

### Link
[helper_player_team_names.py](../whisper_chunk_transcribe/helper_player_team_names.py)

### Description:
This script fetches player and team data from the ESPN API and upserts this information into the database for use in game-related transcription experiments.

### How to Call:
To run the script, use the following command:
```bash
python -m whisper_chunk_transcribe.helper_player_team_names
```

### When to Use:
- **When preparing player and team data for sports-related transcription experiments**.
- **Before running experiments** that require player and team information from specific games, particularly in sports domains like baseball, football, and basketball.

### Summary of Workflow:
- Fetches game data from the ESPN API using `espn_id`.
- Extracts player and team information for the home and away teams.
- Upserts the game, player, and team data into the database for use in experiments.

---

### General Workflow Integration

1. **First**, use `helper_determine_prompt_length_tokens.py` to determine the token lengths of specific prompts, ensuring they fit within the transcription engine's limits.
2. **Next**, use `helper_update_prompt_terms.py` to update and generate valid prompt terms for an experiment, ensuring that the final prompt fits within the token limit before running the actual transcription experiments. 
3. **Finally**, use `helper_player_team_names.py` to fetch and store player/team data for sports-related experiments.

This order ensures that all necessary data (players, teams, prompt terms) is correctly set up before executing the main transcription experiments.