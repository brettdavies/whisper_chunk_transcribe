#!/bin/bash

# 1. Rebuild the whisper_chunk_transcribe package
poetry build -C ~/github/whisper_chunk_transcribe/ 

# Ensure the build command finishes before continuing
if [ $? -ne 0 ]; then
  echo "Build failed, exiting."
  exit 1
fi

# 2. Change to the parent directory (cd ..)
cd ~/Documents

# 3. Delete the existing whisper_chunk_transcribe-0.1.0 directory
rm -rf ~/Documents/whisper_chunk_transcribe-0.1.0

# 4. Deactivate any currently active virtual environment
deactivate

# 5. Activate the virtual environment
source ~/Documents/.venv/bin/activate

# 6. Install the wheel file
pip install ~/github/whisper_chunk_transcribe/dist/whisper_chunk_transcribe-0.1.0-py3-none-any.whl

# 7. Extract the tar.gz file to /home/bigdaddy/Documents
tar -xzvf ~/github/whisper_chunk_transcribe/dist/whisper_chunk_transcribe-0.1.0.tar.gz -C ~/Documents

# 8. Copy the .env file to the extracted directory
cp ~/github/whisper_chunk_transcribe/.env ~/Documents/whisper_chunk_transcribe-0.1.0

# 9. Change to the whisper_chunk_transcribe-0.1.0 directory
cd ~/Documents/whisper_chunk_transcribe-0.1.0

# 10. Print a success message
echo "Tasks completed successfully."
