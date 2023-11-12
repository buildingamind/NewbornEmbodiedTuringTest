# purpose - controlled classification TASK#8 : drop samples from viewpoints

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import Callback


class InvariantRecognition(Dataset):
    def __init__(
        self,
        data_dir: str,
        drop_train_samples: int,
        viewpoints: List[int] = None,
        transform: Optional[Callable] = T.ToTensor(),
        training: bool = False, # extra parameter added
        
    ):
        if viewpoints is None:
            viewpoints = list(range(1, 13))
        self.viewpoints = viewpoints
        self.training = training
        self.drop_train_samples = drop_train_samples
        self.dataframe = self.create_dataframe(data_dir, self.training, self.drop_train_samples)
        self.transform = transform
        

    def create_dataframe(self, data_dir: str, training: bool, drop_train_samples: int) -> pd.DataFrame:
        dfs = []
        subdirs = [f for f in os.scandir(data_dir) if f.is_dir()]
        # Create dataframe for each subdirectory.
        for subdir in subdirs:
            path = subdir.path
            object_name = subdir.name
            viewpoint = int(object_name.split("O")[0][1:])

            # Only include images in the specified viewpoints.
            if viewpoint in self.viewpoints:
                df = pd.DataFrame()
                df["image_path"] = [f.path for f in os.scandir(path) if f.name.endswith(".png")]
                df["object_name"] = object_name
                df["viewpoint"] = viewpoint
                if(object_name[-1]== "1"):
                    df["label"] = 0
                else:
                    df["label"] = 1
                
                #print(df)
                # check if training viewpoints is passed
                if self.training == True:
                    #print("True")
                    '''
                    11 training viewpoint fold (12 fold) - 0 samples to drop
                    8 training viewpoint fold - 9625 samples to drop
                    4 training viewpoint fold - 8250 samples to drop
                    2 training viewpoint fold - 5500 samples to drop
                    1 training viewpoint fold (12sparse) - 0 samples to drop
                    '''
                    df.drop(df.tail(drop_train_samples).index, inplace = True) # changed from drop_train_samples to 0 for tile experiments
                    #print(df)
                    pass
                else:
                    #print("False")
                    
                    df.drop(df.tail(10000).index, inplace = True) # changed from 10k to 0
                    
                    #print(df)
                    pass
                dfs.append(df)
        # Concatenate the dataframes into a single dataframe.
        return pd.concat(dfs, axis=0, ignore_index=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        image_path = self.dataframe["image_path"][idx]
        image = Image.open(image_path)
        label = self.dataframe["label"][idx]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path # image_path is added extra here for the wandb logging table


class InvariantRecognitionDataModule(LightningDataModule):
    name = "invariant_recognition"
    dataset_cls = InvariantRecognition
    dims = (3, 64, 64) # this is used in evaluate.py file
    print("image dims selected in invariant_recognition.py - ", dims)
    viewpoints = list(range(1, 13)) # 1..12

    def __init__(
        self,
        data_dir: str,
        identifier: str,
        num_folds: int = 0,
        val_fold: int = 0, # fold_idx
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_folds: Number of validation folds to create.
            val_fold: Index of validation fold (in range [0, 5])
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.identifier = identifier
        self.num_folds = num_folds
        self.val_fold = val_fold #fold_idx
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.drop_train_samples = 0 # by default, dont drop any train samples

        if self.num_folds > len(self.viewpoints) or self.num_folds < 2:
            raise ValueError(
                    f"num_folds should be smaller than {self.viewpoints} and larger than 2."
            )
        elif self.val_fold >= self.num_folds  or self.val_fold < 0:
            raise ValueError(f"")

    @property
    def default_transforms(self) -> Callable:
        """ Return default data transformation. """
        return T.ToTensor()

    @property
    def viewpoint_splits(self) -> Tuple[List[int], List[int]]:
        """ Split viewpoints into train and validation folds. """
        n_viewpoints = int(len(self.viewpoints) / self.num_folds)
        # for 6 fold (k=6), val_viewpoints = [1,2]
        # and train_viewpoints = [3,4,.....,12] , leave the ones in val_viewpoints

        val_viewpoints = [self.val_fold * n_viewpoints + i + 1 for i in range(n_viewpoints)]
        train_viewpoints = [v for v in self.viewpoints if v not in val_viewpoints]

       
        print("train_viewpoints: ", train_viewpoints)
        print("val_viewpoints: ", val_viewpoints)
        
        #print("graph - 6_fold\n")
        #return train_viewpoints, val_viewpoints #-> 6 fold, for first graph of paper [pass 6 in flags]
        #print("graph - 6_spase(flag=6), 12_sparse(flag=12)\n")
        #return val_viewpoints, train_viewpoints #- > for inverse of 6 fold, second graph of paper [pass 6 in flags]
        
        #print("graph - 1_fold\n")
        #return val_viewpoints, train_viewpoints #- > for inverse of 6 fold, third graph of paper [pass 12 in flags]


        if self.identifier == "6fold":
            print("Fold selected = 6Fold\n") #flag = 6
            #self.drop_train_samples = update when needed
            return train_viewpoints, val_viewpoints

        elif self.identifier == "6sparse":
            print("Fold selected = 6sparse\n") # flag = 6
            self.drop_train_samples = 5500
            return val_viewpoints, train_viewpoints

        elif self.identifier == "12sparse":
            print("Fold selected = 12sparse\n") # flag = 12
            self.drop_train_samples = 0
            return val_viewpoints, train_viewpoints

        elif self.identifier == "12fold": # flag = 12
            print("Fold selected = 12fold\n")
            self.drop_train_samples = 10000
            return train_viewpoints, val_viewpoints

        elif self.identifier == "8fold": # flag = 3
            print("Fold selected = 8 TrainingViewpoints\n")
            self.drop_train_samples = 9625
            return train_viewpoints, val_viewpoints

        elif self.identifier == "4fold": # flag = 3
            print("Fold selected = 4 TrainingViewpoints\n")
            self.drop_train_samples = 8250
            return val_viewpoints, train_viewpoints

        else:
            # implement code to throw error
            print("Wrong fold type or num_folds selected\n")
        


    """
    This is where we will load in data from the file and prepare PyTorch tensor datasets for each split.
    the data split is thus reproducible. This method expects a stage arg which is used to separate logic for 'fit' and 'test'.
    This is helpful if we don't want to load the entire dataset at once.
    The data operations that we want to perform on every GPU is defined here.
    This includes applying transform to the PyTorch tensor dataset.
    """
    
    def setup(self, stage: Optional[str] = None) -> None: # lighteningDataModule prebuilt function
        """ Create train, val, and test dataset. """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms if self.val_transforms is None else self.val_transforms

            train_viewpoints, val_viewpoints = self.viewpoint_splits # returns only two lists.
            self.dataset_train = self.dataset_cls(
                self.data_dir, self.drop_train_samples, viewpoints=train_viewpoints, transform=train_transforms, training=True) # extra argument to know when training view is passed
            # initially it was, viewpoints=val_viewpoints
            self.dataset_val = self.dataset_cls(
                self.data_dir, self.drop_train_samples, viewpoints=train_viewpoints, transform=val_transforms, training=False) # second parameter is the drop_train_samples
            print("len of dataset_train: ", len(self.dataset_train))
            #print("len of dataset_val: ", len(self.dataset_val))

        # stage here defines if trainer.test() is called or trainer.fit() and based on that loads the data.
        if stage == "test" or stage is None:
            test_transforms = self.default_transforms if self.test_transforms is None else self.test_transforms
            val_transforms = self.default_transforms if self.val_transforms is None else self.val_transforms

            # initially, in Don's code, viewpoints arg was not passed in the below line of code.
            self.dataset_test = self.dataset_cls(self.data_dir,0, viewpoints = val_viewpoints ,transform=test_transforms, training=False)

            print("len of dataset_test", len(self.dataset_test))


    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
    
