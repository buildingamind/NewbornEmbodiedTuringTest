import os
import torch, torchvision
from torchvision import transforms as tr
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import pandas as pd
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
        # create a dataframe of training and testing samples 
        self.dataframe = self.create_dataframe(data_dir, self.training, self.drop_train_samples)
        # print(self.dataframe)
        # self.dataframe.to_csv('dataframe.csv', index=False)
        self.transform = transform
        self.image_transform = self._get_transform(image_size=224)
        # create a list of tuples from above dataframe 
        self.path_label_list = self.create_list()
        # print(self.path_label_list)
        # finally, create n_frames sequence dataset for dataloader
        if self.training == True:
            self.n_groupframes = 22000
        else:
            self.n_groupframes = 2000
        
        self.seq_list = self.make_dataset(path_label_list=self.path_label_list, seq_len=16, n_groupframes=self.n_groupframes, ds_rate=1, image_size=224)
        # print(self.seq_list)

        # clean the image_list and label_list from any sequence that has both object1 and object2 inside it
        # do this inside break_tuple function itself.
        self.image_list, self.label_list = self.break_tuple(self.seq_list)
        #print("image list - ", self.image_list)
        #print("label list - ", self.label_list)




    def custom_sort_key(self, entry):
        return int(entry.name.split('O')[1])     
    
    def create_dataframe(self, data_dir: str, training: bool, drop_train_samples: int) -> pd.DataFrame:
        dfs = []
        subdirs = [f for f in os.scandir(data_dir) if f.is_dir()] # [./V1O1, ./V1O2, ....]
        subdirs = sorted(subdirs, key=self.custom_sort_key)
        #print(subdirs)
        # sort the subdirs here and then check the df for training!!

        # Create dataframe for each subdirectory.
        for subdir in subdirs:
            path = subdir.path
            object_name = subdir.name
            viewpoint = int(object_name.split("O")[0][1:])

            # Only include images in the specified viewpoints.
            if viewpoint in self.viewpoints:
                df = pd.DataFrame()
                #df["image_path"] = [f.path for f in os.scandir(path) if f.name.endswith(".png")]
                df["image_path"] = sorted([f.path for f in os.scandir(path) if f.name.endswith(".png")], key=lambda x: int(x.split('_')[-1].split('.')[0]))
                df["object_name"] = object_name
                df["viewpoint"] = viewpoint
                if(object_name[-1]== "1"):
                    df["label"] = 0
                else:
                    df["label"] = 1
                
                #print(df)
                # check if training viewpoints is passed
                if self.training == True:
                    df.drop(df.tail(drop_train_samples).index, inplace = True) # changed from drop_train_samples to 0 for tile experiments.
                    #print(df)
                    
                    pass
                else:
                    
                    df.drop(df.tail(10000).index, inplace = True) # changed from 10k to 0 for tile experiments.
                    #print(df)
                    pass
                dfs.append(df)
                
        # Concatenate the dataframes into a single dataframe.
        return pd.concat(dfs, axis=0, ignore_index=True)

    # AIM: create a tuple of img_paths from df and its label inside a list - 
    def create_list(self):

        image_paths = self.dataframe["image_path"].tolist()

        label = self.dataframe["label"].tolist()
        
        # create a tuple of (image_path, label)
        path_label_list = list(zip(image_paths, label))
        return path_label_list
    
    # AIM: first create n_frames sequences using helper functions, second pass those sequences to dataset module.
    def make_dataset(self, path_label_list, **kwargs):
        seq_len = kwargs['seq_len'] # 16
        n_groupframes=kwargs['n_groupframes']#1450000
        ds_rate = kwargs['ds_rate'] # 1
        image_size = kwargs['image_size']
        #transform = self._get_transform(image_size) # basic normalization, just pass the image size for cropping purposes.
        gx_fpathlist = path_label_list
        n_trainsamples = int(n_groupframes/seq_len) # number of 16 frames sequences in training set, 687 for 11k training samples for a linear probe
        
        if self.training == True:
            print("Number of 16 frames sequences in training dataset :: ", n_trainsamples)
        else:
            print("Number of 16 frames sequences in test dataset :: ", n_trainsamples)
            
        #-----------------
        
        gx_train_fpathseqlist = self.get_fpathseqlist(gx_fpathlist, seq_len, ds_rate=ds_rate, n_samples=n_trainsamples) # creates x sequences of 16 frames each
        return gx_train_fpathseqlist
    
    def break_tuple(self, seq_list):
        image_list, label_list = [],[]
        for sequence in seq_list:
            temp_im, temp_lb = [],[]
            for unit in sequence:
                img, lbl = unit
                temp_im.append(img)
                temp_lb.append(lbl)

            # remove 16 frames sequences which has labels from both : ob1 and ob2 both.
            if self.are_all_elements_same(temp_lb) == 1:
                image_list.append(temp_im)
                label_list.append(temp_lb)
            else:
                continue

        # clean image_list and label_list - 
        
        return image_list, label_list

    # find if there are 16 frames sequences in the dataset that has both object 1 and object 2 labels
    # if set returns len == 1, then all are duplicate values else, different values.
    def are_all_elements_same(self,lst):
        return len(set(lst)) == 1


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load the sequence of images
        # Step4: return both labels and images and then the batch becomes a list.

        # concat images together
        images = torch.cat([
            self.image_transform(torchvision.io.read_image(fp)).unsqueeze(0)
                     for fp in self.image_list[idx]])
        
        labels = torch.cat([
                torch.tensor([int(label)]) for label in self.label_list[idx]])
        
        return images, labels

    def get_fpathseqlist(self, fpathlist, seq_len, ds_rate=1, n_samples=None): #n_samples = total num of sequences which has 16 frames.
        """
        Returns a list of list that can be passed to ImageSequenceDataset: [[(img1, label), (img2, label), ... ,(img16, label)]]
        # n_samples: int
        # between 1 and len(fpathlist)
        # If None, it's set to len(fpathlist)/seq_len
        """
        
        sample_len = seq_len*ds_rate
        if n_samples is None:
            n_samples = int(len(fpathlist)/seq_len)
            sample_stride = sample_len
        else:
            assert type(n_samples)==int
            assert len(fpathlist)>n_samples
            sample_stride = int(len(fpathlist)/n_samples) # 16

        fpathseqlist = [fpathlist[i:i+sample_len:ds_rate] 
                        for i in range(0, n_samples*sample_stride, sample_stride)]
        return fpathseqlist
    
    def _get_transform(self, image_size):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25] 
        
        augs = [tr.Resize(image_size), tr.CenterCrop(image_size), 
                tr.ConvertImageDtype(torch.float32), 
                tr.Normalize(mean,std)]
        return tr.Compose(augs)



class InvariantRecognitionDataModule_Nframes(LightningDataModule):
    name = "invariant_recognition"
    dataset_cls = InvariantRecognition
    dims = (3, 224, 224)
    print(dims)
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
        self.shuffle = shuffle # added on-top
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

        if self.identifier == "6fold":
            print("Fold selected = 6Fold\n") #flag = 6
            #self.drop_train_samples = update when needed
            return train_viewpoints, val_viewpoints

        elif self.identifier == "6sparse":
            print("Fold selected = 6sparse\n")
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

        # stage here defines if trainer.test() is called or trainer.fit() and based on that loads the data.
        if stage == "test" or stage is None:
            test_transforms = self.default_transforms if self.test_transforms is None else self.test_transforms
            val_transforms = self.default_transforms if self.val_transforms is None else self.val_transforms

            # initially, in Don's code, viewpoints arg was not passed in the below line of code.
            self.dataset_test = self.dataset_cls(self.data_dir,0, viewpoints = val_viewpoints ,transform=test_transforms, training=False)


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
    
