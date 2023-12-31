import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/home/minji.tts/run/training/GPT_XTTS_v2.0_LJSpeech_FT-December-31-2023_05+00AM-0000000/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
  config, 
  checkpoint_path="/home/minji.tts/run/training/GPT_XTTS_v2.0_LJSpeech_FT-December-31-2023_05+00AM-0000000/best_model_3920.pth",
  vocab_path="/home/minji.tts/run/training/XTTS_v2.0_original_model_files/vocab.json",
  speaker_file_path="/home/minji.tts/run/training/XTTS_v2.0_original_model_files/model.pth"
)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["/home/minji.tts/MyDataset/wavs/audio1.wav"])

print("Inference...")
out = model.inference(
    "안녕하세요. 김은종입니다.",
    "ko",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)