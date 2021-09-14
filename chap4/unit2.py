import os
import torch
from torch.utils import data
import torchaudio

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def load_audio_files(path:str, label:str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split("_nohash_")
        utterance_number = int(utterance_number)

        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])
    
    return dataset

trainset_speech_commands_yes = load_audio_files('./data/SpeechCommands/speech_commands_v0.02/yes', 'yes')
trainset_speech_commands_no = load_audio_files('./data/SpeechCommands/speech_commands_v0.02/no', 'no')

