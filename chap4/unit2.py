import os
import torch
from torch.utils import data
import torchaudio

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

import matplotlib.pyplot as plt

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

trainloader_yes = data.DataLoader(
    trainset_speech_commands_yes,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


trainloader_no = data.DataLoader(
    trainset_speech_commands_no,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

def show_waveform(waveform, sample_rate, label):
    """
    docstring
    """
    print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, label))
    new_sample_rate = sample_rate / 10
    print(new_sample_rate)

    channel = 0

    waveform_transformed = torchaudio.transforms.Resample(
        sample_rate, new_sample_rate)(waveform[channel,:].view(1,-1)
    )

yes_waveform = trainset_speech_commands_yes[0][0]
yes_sample_rate = trainset_speech_commands_yes[0][1]
no_waveform = trainset_speech_commands_no[0][0]
no_sample_rate = trainset_speech_commands_no[0][1]

show_waveform(yes_waveform, yes_sample_rate, "yes")

def show_spectrogram(waveform):
    spectrogram = torchaudio.transforms.Spectrogram()(waveform)
    #print(spectrogram)
    print("Shape of spectrogram: {}".format(spectrogram.size()))

    plt.figure()
    plt.imsave(f'test/spectrogram_img.png', spectrogram.log2()[0,:,:].numpy(), cmap='gray')

show_spectrogram(yes_waveform)

def show_merspectrogram(waveform, sample_rate):
    """
    docstring
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate
    )(waveform)
    plt.figure()
    plt.imsave(f'test/spectrogram_img.png', mel_spectrogram.log2()[0,:,:].numpy(), cmap='gray')


def show_mfcc(waveform,sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate= sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0,:,:].numpy(), cmap='gray')
    
    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0,:,:].numpy())
    plt.draw()

show_mfcc(no_waveform, no_sample_rate)