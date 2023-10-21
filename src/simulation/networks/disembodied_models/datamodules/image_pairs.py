# FIXME: for 0 division error fix that with some error handling statements inside the drop ep function.


# Method#2 implemented script for paper#2 temporal models
# print the temporal mode while running for a sanity check.

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torchvision.transforms as T
from PIL import Image
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torchvision.datasets import VisionDataset

from .imagefolder_datamodule import ImageFolderDataModule
import torch
import random


class ImagePairs(VisionDataset):
    """ Creates temporally ordered pairs of images from sequnces of visual observations.
    This class assumes each bottom-most directory has a sequence of images with file names:

        root/.../0.png
        root/.../1.png
        ...
        root/.../t.png

    where t is the last timestep in the sequence.

    Args:
        root: Root directory path.
        window_size: Size of sliding window for sampling pairs. If the window_size
            is 1, each sample will return a pair of identical images. Otherwise,
            the samples will consist of all temporally ordered pairs of images
            within the sliding time window.
    """
    def __init__(
        self,
        root: str,
        window_size: int = 3,
        temporal_mode: str = '2+images', # problem, not passing value further - fix this same way as the global variable for drop_ep.
        transform: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(root,
                         transform=transform,
                         target_transform=None)
        self.drop_episodes = dataset_drop
        self.temporal_mode = temporal_mode
        self.window_size = window_size
        self.episodes = self._find_episodes()

        self.total_ep = len(self.episodes)
        if self.drop_episodes != self.total_ep and self.drop_episodes != 0:
            self.episodes = self.episodes[:self.total_ep-self.drop_episodes]

        self.samples = self._make_pairs()
        
        print("number of episodes dropped are - ", self.drop_episodes)
        
        # experimental
        #torch.save(self.samples, '2+images_in_a_window_ws=3.pt')

    def _find_episodes(self) -> List[str]:
        """ Find paths to bottom-most directories containing image sequences."""
        episode_paths = []
        for path, dirs, files in sorted(os.walk(self.root)):
            # Only consider the bottom-most directories.
            if not len(dirs) and len(files) > 0:
                episode_paths.append(path)
        
        # newly added line to sort the episodes
        episode_paths.sort(key=lambda path: int(path.split("ep")[1]))
        return episode_paths

    # # drop dataset episodes if you don't want to use the entire dataset
    # def drop_ep(self,):
    #     if self.dataset_size > 0 and self.ep_samples > 0:
    #         num_drop_ep = self.dataset_size // self.ep_samples # 40000
    #     #print("num_drop_ep ", num_drop_ep)
    #         return num_drop_ep
    

    def _make_pairs(self) -> List[Tuple[str, str]]:
        # push 2 images together in a window:
        if self.temporal_mode  == '2images':
            print("Pushing two images in the temporal window!")

            pairs = []
            for episode in self.episodes: 
            # Sort file names numerically in ascending order.
                fnames = sorted(
                    [d.name for d in os.scandir(episode) if d.is_file()], 
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[1]) 
                )

                # shuffling samples 
                #random.shuffle(fnames)

            # Sample pairs with sliding time window.
                for i in range(len(fnames) - self.window_size):               
                    prev_path = os.path.join(episode, fnames[i])    
                    next_path = os.path.join(episode, fnames[i+self.window_size-1])
                    pairs.append((prev_path, next_path))   
            #print(pairs)    
            return pairs

        # push more than 2 images in a window
        elif self.temporal_mode == '2+images':
            print("Pushing 2+ images in the temporal window!")
            pairs = []
            for episode in self.episodes: 
            # Sort file names numerically in ascending order.
            # fnames has the images from the episodes.
                fnames = sorted(
                    [d.name for d in os.scandir(episode) if d.is_file()], 
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[1]) 
                )
            # Sample pairs with sliding time window.
                for i in range(0, len(fnames)-self.window_size+1):
                    temp = []
                    for j in range(i,i+self.window_size):                
                        path = os.path.join(episode, fnames[j])
                        temp.append(path)
                    
                    # [[1,2,3,4],[2,3,4,5],...]
                    pairs.append(temp) 
            #print(pairs)
            return pairs

    
    def __getitem__(self, index: int):
        # if 2 images in a temporal window - 
        if self.temporal_mode == "2images":
            prev_path, next_path = self.samples[index]
            prev_img = Image.open(prev_path) # PIL format
            next_img = Image.open(next_path) # PIL format

            if self.transform is not None:
                prev_img = self.transform(prev_img)
                next_img = self.transform(next_img)

            return prev_img, next_img, index 
        # more than 2 images in a temporal window -
        else:
            sample_list = self.samples[index]
        
            # transform samples from list
            for i in range(0,len(sample_list)):
                sample_list[i] = Image.open(sample_list[i])
                if self.transform is not None:
                    sample_list[i] = self.transform(sample_list[i])
            
            # current temporal support for 3 and 4 images per window
            if len(sample_list) == 3:
                return sample_list[0], sample_list[1], sample_list[2], index
            else:
                return sample_list[0], sample_list[1], sample_list[2], sample_list[3], index

    def __len__(self) -> int:
        return len(self.samples)

# global variable - 
dataset_drop = 0


class ImagePairsDataModule(VisionDataModule):
    name = "image_pairs"
    dataset_cls = ImagePairs # this is alias of pl_bolts.datasets.emnist_dataset.BinaryEMNIST
    dims = (3, 64, 64)
    #dims = (3, 224, 224)

    print("image dims selected in image_pairs.py - ", dims)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        window_size: int = 3,
        temporal_mode: str = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        drop_ep: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,

            *args,
            **kwargs,
        )
        
        # make a global variable
        global dataset_drop
        # assign value to that global variable
        dataset_drop = drop_ep
        
        self.temporal_mode = temporal_mode
        self.window_size = window_size
        self.EXTRA_ARGS = {"window_size": window_size}
        print('Inside Temporal Datamodule')

    @property
    def num_samples(self):
        """ Number of training samples. """
        dataset = self.dataset_cls(
            root=self.data_dir,
            window_size=self.window_size, 
            temporal_mode=self.temporal_mode,  # temporal_mode flag for selecting images to push in a window
            ) 
        print("global variable value is - ", dataset_drop)
        print(dataset)

        
        # Implicitly takes of splits using VisionDataModule function
        return len(self._split_dataset(dataset))

    def default_transforms(self) -> Callable:
        return T.ToTensor()
    
