import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# input fs
CONFIG_PATH = "run/training/GPT_XTTS_Carl_20251002-October-03-2025_11+12AM-37eb147/config.json"
TOKENIZER_PATH = "run/training/XTTS_v2_original_model_files/vocab.json"
XTTS_CHECKPOINT = "run/training/GPT_XTTS_Carl_20251002-October-03-2025_11+12AM-37eb147/best_model.pth"
SPEAKER_REFERENCE = "C:/Users/monol/wkdir/code/llm_tts/wav/input/Carl_Voice_Dataset/wavs/carl_0001.wav"

# output wav path
OUTPUT_WAV_PATH = "test_carl_20251002.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "Check it out Fryman, I got a voice now. It's me, car Carl.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)