import asyncio
from faster_whisper import WhisperModel
from .transcription_processor import TranscriptionProcessor
from .database import DatabaseOperations

async def main() -> None:
    # Initialize the Whisper model
    model = WhisperModel('/media/bigdaddy/data/cache_model/faster-distil-whisper-medium.en')  # Replace 'model_name' with your actual model name

    # Initialize the database operations stub class
    db_ops = DatabaseOperations()

    # Create an instance of TranscriptionProcessor
    processor = TranscriptionProcessor(worker_name="TestWorker", model=model, db_ops=db_ops)

    # prompt_terms = db_ops.get_prompt_terms()
    # for prompt in prompt_terms:
    #     token_length = await processor.determine_token_length(prompt)
    #     db_ops.set_prompt_token_length(prompt, token_length)
    #     print(f"Prompt: {prompt} has {token_length} tokens")

    prompt = "terms: "
    prompt_length = await processor.determine_token_length(prompt)
    print(f"Prompt: {prompt} has {prompt_length} tokens")
    
    prompt_terms = db_ops.get_exp_experiment_prompt_terms(2)
    print(f"Prompt terms: {prompt_terms}")

    first_term = True
    for term in prompt_terms:
        term_length = await processor.determine_token_length(term)
        # Check if adding the term would exceed the length limit
        if (prompt_length + term_length) <= 255:
            if first_term:
                prompt += f"{term}"  # Append the first term without a preceding comma
                first_term = False  # Set the flag to False after adding the first term
            else:
                prompt += f", {term}"  # Append subsequent terms with a preceding comma
            prompt_length = await processor.determine_token_length(prompt)  # Update the prompt_length
        else:
            pass
    
    print(f"{prompt_length} tokens in prompt \"{prompt}\"")

if __name__ == "__main__":
    # Run the main function with the provided prompt
    asyncio.run(main())

# Run the script as a Module:
# python -m whisper_chunk_transcribe.update_prompt_terms
