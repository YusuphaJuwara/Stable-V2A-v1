import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Union
from main.data.dataset_gh import GreatestHitsDataset
import numpy as np

############################################################################
# DATA MODULE
############################################################################

class GreatestHitsDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_split_file_path: str,
        train_data_to_use: float,
        train_frames_transforms: Union[transforms.Compose, None],
        val_split_file_path: str,
        val_data_to_use: float,
        val_frames_transforms: Union[transforms.Compose, None],
        test_split_file_path: str,
        test_data_to_use: float,
        test_frames_transforms: Union[transforms.Compose, None],
        chunk_length_in_seconds: float,
        sr: int,
        audio_file_suffix: str,
        annotations_file_suffix: str,
        metadata_file_suffix: str,
        frame_file_suffix: str,
        force_channels: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        rms_path: str,

    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.train_split_file_path = train_split_file_path
        self.train_data_to_use = train_data_to_use
        self.train_frames_transforms = train_frames_transforms
        self.val_split_file_path = val_split_file_path
        self.val_data_to_use = val_data_to_use
        self.val_frames_transforms = val_frames_transforms
        self.test_split_file_path = test_split_file_path
        self.test_data_to_use = test_data_to_use
        self.test_frames_transforms = test_frames_transforms
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.sr = sr
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.force_channels = force_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.rms_path = rms_path
    
    def collate_fn(self, batch):
        # Unpack the batch
        audios, frames, seconds_start, seconds_total, chunk_rms, items = zip(*batch)

        # Pad audios to have shape [2, self.chunk_length_in_seconds * self.sr]
        sample_size = int(self.chunk_length_in_seconds * self.sr)
        audios_padded = []
        for audio in audios:
            if audio.shape[1] > sample_size:
                # Truncate the audio
                audio = audio[:, :sample_size]
            elif audio.shape[1] < sample_size:
            # Pad the audio
                audio = F.pad(audio, (0, sample_size - audio.shape[1]))
            audios_padded.append(audio)
        # Stack audios and frames
        audios = torch.stack(audios_padded)
        frames = torch.stack(frames)
        chunk_rms = torch.stack(chunk_rms)


        # Combine items into a single list
        items = [item for sublist in items for item in sublist]

        return audios, frames, seconds_start, seconds_total, chunk_rms, items

    def setup(self, stage) -> None:
        if stage == "fit" or stage == "validate":
            self.train_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.train_split_file_path,
                split='train',
                data_to_use=self.train_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sr=self.sr,
                frames_transforms=self.train_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
                force_channels=self.force_channels,
                rms_path=self.rms_path,
            )

            self.val_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.val_split_file_path,
                split='val',
                data_to_use=self.val_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sr=self.sr,
                frames_transforms=self.val_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
                force_channels=self.force_channels,
                rms_path=self.rms_path,
            )

            self.train_dataset.print()
            self.val_dataset.print()

        if stage == "test":
            self.test_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.test_split_file_path,
                split='test',
                data_to_use=self.test_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sr=self.sr,
                frames_transforms=self.test_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
                force_channels=self.force_channels,
                rms_path=self.rms_path,
            )

            self.test_dataset.print()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )


if __name__ == '__main__':
    from torchvision import transforms

    train_transforms = [
        transforms.Resize((240, 240), antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_transforms = [
        transforms.Resize((240, 240), antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    test_transforms = [
        transforms.Resize((240, 240), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    datamodule = GreatestHitsDatamodule(
        root_dir="GREATEST-HITS-DATASET/mic-mp4-processed/",
        train_split_file_path="GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        train_data_to_use=0.01,
        train_frames_transforms=train_transforms,

        val_split_file_path="GREATEST-HITS-DATASET/mic-mp4/val.txt",
        val_data_to_use=0.1,
        val_frames_transforms=val_transforms,

        test_split_file_path="GREATEST-HITS-DATASET/mic-mp4-processed/test.txt",
        test_data_to_use=0.1,
        test_frames_transforms=test_transforms,

        chunk_length_in_seconds=2.0,
        sr=44100,
        force_channels="stereo",

        audio_file_suffix=".resampled.wav",
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )