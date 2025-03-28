import torch
from TTS.api import TTS

# Check if CUDA is installed
if torch.cuda.is_available():
    print("CUDA installed successfully\n") 
else:
    print("CUDA not properly installed. Stopping process...")
    quit()

# Print available TTS models
view_models = input("View models? [y/n]\n")
if view_models == "y":
    tts_manager = TTS().list_models()
    all_models = tts_manager.list_models()
    print("TTS models:\n", all_models, "\n", sep = "")

# Prompt model selection
model = input("Enter model:\n")
# for example, tts_models/multilingual/multi-dataset/xtts_v2

# Example voice cloning with selected model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tts = TTS((model), progress_bar=True).to(device)
tts.tts_to_file("This is a voice cloning test - this robot stole my voice!", speaker_wav="C:\\Users\\monol\\wkdir\\llm_tts\\train-audio.wav",
                language="en", file_path="C:\\Users\\monol\\wkdir\\llm_tts\\output.wav")