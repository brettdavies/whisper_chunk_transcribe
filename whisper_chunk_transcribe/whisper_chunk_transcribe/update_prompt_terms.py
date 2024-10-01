# Standard Libraries
import asyncio
import argparse
from typing import List
from dataclasses import dataclass

# Third Party Libraries
from faster_whisper import WhisperModel

# First Party Libraries
from .transcription_processor import TranscriptionProcessor
from .database import DatabaseOperations

@dataclass
class ExperimentConfig:
    experiment_id: int
    max_tokens: int = 255  # Default value

class ExtendedDatabaseOperations(DatabaseOperations):
    def get_exp_experiment_prompt_terms(self, experiment_id: int) -> List[str]:
        """
        Retrieves the prompt terms for a given experiment ID from the database.

        Args:
            experiment_id (int): The ID of the experiment.

        Returns:
            List[str]: A list of prompt terms.

        Raises:
            Exception: If there is an error executing the query.
        """
        select_query = """
            SELECT term
            FROM exp_experiment_prompt_terms
            WHERE experiment_id = %s
            ORDER BY final_score DESC;
        """
        print("Fetching prompt terms from the database.")

        # Execute the query
        params = (experiment_id,)
        records = self.execute_query("TestWorker", select_query, params)

        # Extract 'term' from each record
        prompt_terms = [record[0] for record in records]
            
        print(f"Retrieved {len(prompt_terms)} prompt terms.")
        return prompt_terms

async def main(config: ExperimentConfig) -> None:
    """
    Main function for updating prompt terms.
    
    Args:
        config (ExperimentConfig): The experiment configuration.
    
    Returns:
        None
    """
    # Initialize the Whisper model
    model = WhisperModel('/media/bigdaddy/data/cache_model/faster-distil-whisper-medium.en')  # Replace 'model_name' with your actual model name

    # Initialize the database operations stub class
    db_ops = ExtendedDatabaseOperations()

    # Create an instance of TranscriptionProcessor
    processor = TranscriptionProcessor(worker_name="TestWorker", model=model, db_ops=db_ops)

    prompt = "terms: "
    prompt_length = await processor.determine_token_length(prompt)
    print(f"Prompt: {prompt} has {prompt_length} tokens")
    
    prompt_terms = db_ops.get_exp_experiment_prompt_terms(config.experiment_id)
    print(f"Prompt terms: {prompt_terms}")

    first_term = True
    for term in prompt_terms:
        term_length = await processor.determine_token_length(term)
        # Check if adding the term would exceed the length limit
        if (prompt_length + term_length) <= config.max_tokens:
            if first_term:
                prompt += f"{term}"  # Append the first term without a preceding comma
                first_term = False  # Set the flag to False after adding the first term
            else:
                prompt += f", {term}"  # Append subsequent terms with a preceding comma
            prompt_length = await processor.determine_token_length(prompt)  # Update the prompt_length
        else:
            pass
    
    print(f"{prompt_length} tokens in prompt \"{prompt}\"")

def valid_positive_int(value: str) -> int:
    """
    Validates that the provided argument is a positive integer.

    Args:
        value (str): The argument value to validate.

    Returns:
        int: The validated positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'.")

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be a positive integer. Got {ivalue}.")
    return ivalue

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run an experiment with specified parameters."
    )

    # --experiment_id as a required argument with type validation
    parser.add_argument(
        "--experiment_id",
        type=valid_positive_int,  # Ensures the input is a positive integer
        required=True,
        help="The experiment ID to use (must be a positive integer)."
    )

    # --max-tokens as an optional named argument with integer validation and default value
    parser.add_argument(
        "--max_tokens",
        type=valid_positive_int,  # Ensures the input is a positive integer
        default=255,
        help="The maximum number of tokens in the prompt (must be a positive integer)."
    )

    # Parse the arguments
    args = parser.parse_args()

    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        max_tokens=args.max_tokens
    )

    # Run the main function with the provided arguments
    asyncio.run(main(config))

# Run the script as a Module:
# python -m whisper_chunk_transcribe.update_prompt_terms
