# Experiment Data Population Script

### Description:
This SQL script contains three statements that populate the database with data required to run transcription experiments. The script:
1. Creates a new experiment.
2. Populates experiment groups by selecting a random 1% of videos.
3. Associates audio segments with the experiment.
4. Calculates prompt scores based on weighted factors.
5. Inserts test cases and assigns them to experiment segments.
6. Evaluates the distribution of segments across test cases to identify any biases.

---

### Statement 1: Create a New Experiment and Populate Experiment Groups

**Purpose**:
- Create a new experiment entry in the `exp_experiments` table.
- Select a random 1% of videos from the `yt_metadata` and `yt_video_file` tables.
- Link the selected videos with the experiment and insert their corresponding audio segments into the `exp_experiment_segments` table.

**Steps**:
1. **Define Date Range**: Specify the range of dates for selecting videos.
2. **Insert Experiment Record**: Create a new experiment record and retrieve its `experiment_id`.
3. **Select and Insert Videos**: Randomly select 1% of the videos and associate them with the experiment.
4. **Insert Audio Segments**: Link the corresponding audio segments of the selected videos with the experiment.

---

### Statement 2: Insert Test Cases and Prompts

**Purpose**:
- Insert test cases for the created experiment, each representing different transcription scenarios, and assign prompt templates.
- Included test cases include:
  - **No Prompt**: Transcribe audio without any additional context.
  - **Unrelated Prompt**: Assess transcription accuracy with an unrelated prompt.
  - **Generic Sports Prompt**: Provide a general sports-related prompt without specific sports terminology.
  - **Baseball-Specific Prompts**: Include both general and game-specific baseball terminology to measure context-based transcription accuracy.
  - **Previous Segment Transcription**: Use the previous segment’s transcription as a prompt.
  
**Steps**:
1. **Write Test Cases**: Modify the script to reflect the test cases you want to run. Use the helper scripts to assist with `prompt_template` and `prompt_tokens`.

    Determine the length of a given prompt ([Script](../whisper_chunk_transcribe/helper_determine_prompt_length_tokens.py))([Docs](helper_scripts.md)):
    ```python 
    python -m whisper_chunk_transcribe.helper_determine_prompt_length_tokens "{prompt_template}"
    ```
    Returns a maximum length prompt based on the prompt term scoring performed in Statement 1 ([Script](../whisper_chunk_transcribe/helper_update_prompt_terms.py))([Docs](helper_scripts.md)):
    ```python 
    python -m whisper_chunk_transcribe.helper_update_prompt_terms
    ```
2. **Insert Test Cases**: Insert various test cases, each with a different prompt and configuration (static or dynamic prompts).
3. **Insert Non-self referencing Prompts**: Prepopulate the test prompts that do not rely on previous segment transcriptions.

---

### Statement 3: Assign Test Cases to Treatment Groups

**Purpose**:
- Randomly assign test cases to the experiment segments created in Statement 1.

**Steps**:
1. **Random Assignment**: For each experiment segment, randomly assign a test case from the `exp_test_cases` table.
2. **Update Experiment Segments**: Update the `exp_experiment_segments` table to reflect the assigned test case for each segment.

---

### Statement 4: Calculate and Evaluate Test Case Distribution

**Purpose**:
- Calculate the distribution of segments across test cases to detect any bias in the assignment of test cases.
- This step is critical to ensuring that the distribution of test cases is balanced, reducing the risk of biased results.

**Steps**:
1. **Count Segments per Test Case**: Calculate the number of segments assigned to each test case.
2. **Calculate Distribution Bias**: Compare the actual segment counts with the expected distribution, determining if a test case is over- or under-represented.
3. **Output Distribution**: Display the distribution for each test case, along with a bias classification:
   - **Over Represented**: More segments than expected.
   - **Under Represented**: Fewer segments than expected.
   - **Perfectly Distributed**: Test case assignment is balanced.

---

### Scoring Logic

**Purpose**:
- Calculate prompt scores based on weighted factors for different categories:
  - **In-Game Usage**: Relevance to gameplay.
  - **General Speech**: Applicability to general speech scenarios.
  - **Impact on Transcription**: How the prompt affects transcription accuracy.
  - **Confusion Potential**: Likelihood of confusing the transcription engine.

**Steps**:
1. **Weight Configuration**: Define weights for each category (modifiable).
2. **Intermediate Results**: Multiply base scores by category weights to compute weighted scores.
3. **Final Score Calculation**: Combine the weighted scores to produce a final prompt score.

---

### Example Query Results

#### Distribution of Segments Across Test Cases:

| Test Case ID | Segment Count | Percentage of Total | Deviation from Perfect Distribution | Bias              |
|--------------|---------------|---------------------|-------------------------------------|-------------------|
| 1            | 100           | 25.00%              | 0                                   | Perfectly Distributed |
| 2            | 90            | 22.50%              | -10                                 | Under Represented    |
| 3            | 110           | 27.50%              | +10                                 | Over Represented     |

This output helps identify any biases in the random assignment of test cases, ensuring a fair and balanced experiment setup.

---

### Usage

Run the entire SQL script to:
1. Prepopulate the database with experiment data, including test cases, videos, audio segments, and prompt terms.
2. Automatically assign test cases to the experiment segments.
3. Analyze the distribution of segments across test cases to ensure experimental balance.

Make sure to modify category weights and other parameters as needed for specific experiment requirements.