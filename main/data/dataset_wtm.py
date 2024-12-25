import torch
import os
import json
import pandas as pd
import random
import glob
import torchvision.transforms as transforms
from natsort import natsorted
from tqdm import tqdm
import torchaudio
import pickle
import numpy as np

import librosa

from PIL import Image

from stable_audio_tools.data.utils import Stereo, Mono, PhaseFlipper
from main.module_controlnet import window_rms, low_pass_filter


class WalkingTheMapsDataset(torch.utils.data.Dataset):
    """
    Dataset to train onset detection model on WalkingTheMaps dataset. 
    Split videos into chunks of N seconds.
    Annotate each chunk with onset labels (1 if onset, 0 otherwise) for each video frame.
    """

    def __init__(
        self,
        root_dir,
        split_file_path,
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=8.0,
        frames_transforms=None,
        sr=16000,
        audio_file_suffix='.wav',
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        force_channels="stereo"
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_file_path = split_file_path
        self.split = split
        self.data_to_use = data_to_use
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.sr = sr
        self.audio_file_suffix = audio_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.force_channels = force_channels

        if frames_transforms is not None:
            self.frames_transforms = frames_transforms
        else:
            self.frames_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )


        # load list chunks from pickle if exists
        # if os.path.exists(f"{self.root_dir}/list_chunks_{self.split}.pkl"):
        #     with open(f"{self.root_dir}/list_chunks_{self.split}.pkl", "rb") as f:
        #         self.list_chunks = pickle.load(f)
        
        # else: 
        # read list of samples
        with open(split_file_path, "r") as f:
            self.list_samples = f.read().splitlines()

        # subset
        if data_to_use < 1.0:
            # shuffle list
            random.shuffle(self.list_samples)
            self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]
            self.list_samples = natsorted(self.list_samples)  # natural sorting (e.g. 1, 2, 10 instead of 1, 10, 2)

        self.list_chunks = []
        self.total_time_in_minutes = 0.0

        for sample in tqdm(self.list_samples, total=len(self.list_samples), desc=f"Loading {split} dataset"):
            # get metadata
            
            metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            # frame_rate = metadata["processed"]["video_frame_rate"]
            frame_rate = 4
            duration = metadata["processed"]["video_duration"]

            # compute number of chunks or set it manually
            num_chunks = int(duration / chunk_length_in_seconds)
            # num_chunks = 1

            end_time = num_chunks * chunk_length_in_seconds

            audio = os.path.join(root_dir, sample, "audio", f"{sample}{audio_file_suffix}")
            audio, sr_native = torchaudio.load(audio)
            if sr_native != self.sr:
                audio = torchaudio.functional.resample(audio, orig_freq=sr_native, new_freq=self.sr)
            # audio, sr_native = torchaudio.load(audio)
            # audio = torchaudio.functional.resample(audio, orig_freq=sr_native, new_freq=self.sr)
            # # Normalize audio amplitude between -1 and 1
            # audio = audio / torch.max(torch.abs(audio))

            # audio, sr = librosa.load(audio, sr=self.sr)
            # Normalize audio amplitude between -1 and 1
            # audio = audio / np.max(np.abs(audio))
            # audio = 2 * (audio - np.min(audio)) / (np.max(audio) - np.min(audio)) - 1.
            

            # get annotations (onsets) up to end time
            # annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
            # annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
            # onset_times = annotations["times"].values
            # onset_times = onset_times[onset_times < end_time]

            self.total_time_in_minutes += end_time

            # get chunks
            chunk_length_in_frames = int(chunk_length_in_seconds * frame_rate)
            for i in range(num_chunks):
                # chunk start and end
                chunk_start_time = i * chunk_length_in_seconds
                chunk_end_time = chunk_start_time + chunk_length_in_seconds
                chunk_start_frame = int(chunk_start_time * frame_rate)
                chunk_end_frame = int(chunk_end_time * frame_rate)
                chunk_start_sr = int(chunk_start_time * self.sr)
                chunk_end_sr = int(chunk_end_time * self.sr)

                audio_chunk = audio[:, chunk_start_sr:chunk_end_sr]
                if self.augs is not None:
                    audio_chunk = self.augs(audio_chunk)
                # Normalize audio amplitude between -1 and 1
                audio_chunk = audio_chunk / torch.max(torch.abs(audio_chunk))
                # audio_chunk = audio_chunk.clamp(-1,1)
                if self.encoding is not None:
                    audio_chunk = self.encoding(audio_chunk)

                # extract onset times for this chunk
                # chunk_onsets_times = annotations[(annotations["times"] >= chunk_start_time) & (annotations["times"] < chunk_end_time)]["times"].values

                # normalize to chunk start
                # chunk_onsets_times = chunk_onsets_times - chunk_start_time

                # convert onset times to frames
                # chunk_onsets_frames = (chunk_onsets_times * frame_rate).astype(int)

                # # compute frames labels
                # labels = torch.zeros(chunk_length_in_frames)

                # # check if chunk_onsets_frames has any index out of labels size
                # max_index = chunk_onsets_frames.max().item() if chunk_onsets_frames.size > 0 else 0
                # if max_index >= labels.size(0):
                #     # extend labels with zeros to fit the max index
                #     labels = torch.cat((labels, torch.zeros(max_index - labels.size(0) + 1)))

                # labels[chunk_onsets_frames] = 1

                # append chunk
                self.list_chunks.append({
                    "video_name": sample,
                    "frames_path": os.path.join(root_dir, sample, "frames_4fps"),
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "start_frame": chunk_start_frame,
                    "end_frame": chunk_end_frame,
                    "audio": audio_chunk,
                    # "labels": labels,
                    "frame_rate": frame_rate,
                    "sample_rate": self.sr,
                    "seconds_start": 0.0,
                    "seconds_total": self.chunk_length_in_seconds,
                })
            
        # self.total_time_in_minutes /= 60.0
        # with open(f"{self.root_dir}/list_chunks_{self.split}.pkl", "wb") as f:
        #     pickle.dump(self.list_chunks, f)
    
    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, index):
        try:
            chunk = self.list_chunks[index]
            frames_list = glob.glob(f"{chunk['frames_path']}/*{self.frame_file_suffix}")
            frames_list = natsorted(frames_list)

            # get frames
            frames_list = frames_list[chunk["start_frame"]:chunk["end_frame"]]
            frames = self.read_image_and_apply_transforms(frames_list)

            # imgs = chunk["imgs"]

            
            # # # get labels
            # # labels = chunk["labels"]

            # # get audio
            audio = chunk["audio"]

            #get features
            # features = chunk["features"]

            seconds_start = chunk["seconds_start"]
            seconds_total = chunk["seconds_total"]

            #get rms
            # rms = chunk["rms"]
            # mu_expanded = chunk["rms_expanded"]
            # rms_original = chunk["rms_original"]

            path_all_rms = "logs/video2rms/rms_pred_WTM_best/"
            with open(f"{path_all_rms}/{chunk['video_name']}_{int(chunk['start_time'])*30}_{int(chunk['end_time'])*30}.npy", "rb") as f:
                chunk_rms = np.load(f)

            chunk_rms = torch.tensor(chunk_rms)

            item = {
                "video_name": chunk["video_name"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "start_frame": chunk["start_frame"],
                "end_frame": chunk["end_frame"],
                # "frames": imgs,
                # "label": labels,
                # "audio": audio,
                # "rms": rms,
                # "rms_expanded": mu_expanded,
                # "rms_original": rms_original,
                # "features": features,
                "text": "A person walks.",
                "frame_rate": chunk["frame_rate"],
                "sample_rate": chunk["sample_rate"],
            }

        except Exception as e:
            print(f"Error reading frames for chunk {index}: {e}")
            print(chunk)
            print('---')
            return self.__getitem__(index + 1)
        
        return audio, frames, seconds_start, seconds_total, chunk_rms, [item]

    def read_image_and_apply_transforms(self, frame_list):
        imgs = []
        # convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            # image = convert_tensor(image)
            if self.frames_transforms is not None:
                image = self.frames_transforms(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        imgs = torch.cat(imgs, dim=0).squeeze()
        # if self.frames_transforms is not None:
        #     imgs = self.frames_transforms(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs

    def print(self):
        print(f"\nWalking The Maps {self.split} dataset:")
        # print(f"num {self.split} samples: {len(self.list_samples)}")
        print(f"num {self.split} chunks: {len(self.list_chunks)}")
        audio, frames, seconds_start, seconds_total, chunk_rms, item = self[0]
        print(f"chunk frames size: {frames.shape}")
        print(f"chunk audio size: {audio.shape}")

if __name__ == '__main__':
    dataset = WalkingTheMapsDataset(
        root_dir="data/Walking-the-maps/videoclip_processsed",
        split_file_path="main/dataset_wtm.py",
        split='train',
        data_to_use=0.01,
        chunk_length_in_seconds=2.0,
        frames_transforms=[
            transforms.ToTensor(),
            transforms.Resize((240, 240), antialias=True),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        audio_file_suffix=".wav",
        # annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        # feature_file_suffix=".pkl"
    )
    dataset.print()


# class GreatestHitsDataset(torch.utils.data.Dataset):
#     """
#     Dataset to train onset detection model on Greatest Hits dataset.
#     Split videos into chunks of N seconds (chunks start at integer second before first onset).
#     Annotate each chunk with onset labels (1 if onset, 0 otherwise) for each video frame
#     """

#     def __init__(
#         self,
#         root_dir,
#         split_file_path,
#         split='train',
#         data_to_use=1.0,
#         chunk_length_in_seconds=2.0,
#         audio_file_suffix='.resampled.wav',
#         annotations_file_suffix=".times.csv",
#         metadata_file_suffix=".metadata.json"
#     ):
#         super().__init__()
#         self.root_dir = root_dir
#         self.split_file_path = split_file_path
#         self.split = split
#         self.data_to_use = data_to_use
#         self.chunk_length_in_seconds = chunk_length_in_seconds
#         self.audio_file_suffix = audio_file_suffix
#         self.annotations_file_suffix = annotations_file_suffix
#         self.metadata_file_suffix = metadata_file_suffix

#         self.video_transform = transforms.Compose(
#             self.generate_video_transform()
#         )

#         with open(split_file_path, "r") as f:
#             self.list_samples = f.read().splitlines()

#         if data_to_use < 1.0:
#             # shuffle list
#             random.shuffle(self.list_samples)
#             self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]

#         self.list_chunks = []
#         self.total_time_in_minutes = 0.0

#         for sample in self.list_samples:
#             # get metadata
#             metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
#             with open(metadata_path, "r") as f:
#                 metadata = json.load(f)
#             frame_rate = metadata["processed"]["video_frame_rate"]

#             # get annotations
#             annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
#             annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
#             onset_times = annotations["times"].values
#             start_time = int(onset_times[0])  # start at integer second before first hit
#             end_time = onset_times[-1] - (onset_times[-1] - start_time) % chunk_length_in_seconds

#             self.total_time_in_minutes += end_time - start_time

#             # get chunks
#             num_chunks = int((end_time - start_time) / chunk_length_in_seconds)
#             chunk_length_in_frames = int(chunk_length_in_seconds * frame_rate)
#             for i in range(num_chunks):
#                 # chunk start and end
#                 chunk_start_time = start_time + i * chunk_length_in_seconds
#                 chunk_end_time = chunk_start_time + chunk_length_in_seconds
#                 chunk_start_frame = int(chunk_start_time * frame_rate)
#                 chunk_end_frame = int(chunk_end_time * frame_rate)

#                 # extract onset times for this chunk
#                 chunk_onsets_labels = annotations[(annotations["times"] >= chunk_start_time) & (annotations["times"] < chunk_end_time)]["times"].values

#                 # normalize to chunk start
#                 chunk_onsets_labels = chunk_onsets_labels - chunk_start_time

#                 # convert onset times to frames
#                 chunk_onsets_frames = (chunk_onsets_labels * frame_rate).astype(int)

#                 # compute frames labels
#                 labels = torch.zeros(chunk_length_in_frames)
#                 labels[chunk_onsets_frames] = 1

#                 # append chunk
#                 self.list_chunks.append({
#                     "frames_path": os.path.join(root_dir, sample, "frames"),
#                     "start_frame": chunk_start_frame,
#                     "end_frame": chunk_end_frame,
#                     "labels": labels
#                 })

#         self.total_time_in_minutes /= 60.0

#     def __getitem__(self, index):

#         chunk = self.list_chunks[index]
#         frames_list = glob.glob(f"{chunk['frames_path']}/*.jpg")
#         frames_list.sort()

#         # get frames
#         frames_list = frames_list[chunk["start_frame"]:chunk["end_frame"]]
#         imgs = self.read_image(frames_list)

#         # get labels
#         labels = chunk["labels"]

#         batch = {
#             'frames': imgs,
#             'label': labels
#         }

#         return batch

#     def __len__(self):
#         return len(self.list_chunks)

#     def read_image(self, frame_list):
#         imgs = []
#         convert_tensor = transforms.ToTensor()
#         for img_path in frame_list:
#             image = Image.open(img_path).convert('RGB')
#             image = convert_tensor(image)
#             imgs.append(image.unsqueeze(0))
#         # (T, C, H ,W)
#         imgs = torch.cat(imgs, dim=0).squeeze()
#         imgs = self.video_transform(imgs)
#         imgs = imgs.permute(1, 0, 2, 3)
#         # (C, T, H ,W)
#         return imgs

#     def generate_video_transform(self):
#         vision_transform_list = []

#         vision_transform_list.append(transforms.Resize((128, 128), antialias=True))

#         if self.split == 'train':
#             vision_transform_list.append(transforms.RandomCrop((112, 112)))
#             vision_transform_list.append(transforms.ColorJitter(
#                 brightness=0.1, contrast=0.1, saturation=0, hue=0
#             ))
#         else:
#             vision_transform_list.append(transforms.CenterCrop((112, 112)))
#             # color_funct = transforms.Lambda(lambda img: img)

#         vision_transform_list.append(transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         ))

#         # vision_transform_list = [
#         #     resize_funct,
#         #     crop_funct,
#         #     color_funct,
#         #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # ]

#         return vision_transform_list

#     def print(self):
#         print(f"\nGreatesthit {self.split} dataset:")
#         print(f"num {self.split} samples: {len(self.list_samples)}")
#         print(f"num {self.split} chunks: {len(self.list_chunks)}")
#         print(f"total time in minutes: {self.total_time_in_minutes}")
#         print(f"chunk frames size: {self[0]['frames'].shape}")
#         print(f"chunk label size: {self[0]['label'].shape}")
