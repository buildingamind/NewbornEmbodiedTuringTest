#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import os
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import torch
from parsing_dataset import ParsingDataset
from sklearn.manifold import TSNE
from matplotlib import cm

class VisualizeEncoder(object):
    
    def __init__(self, model, save_path, vector_length, dvs_wrapper,data_path = "./data") -> None:
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(ParsingDataset(Path(data_path),dvs_wrapper), batch_size=1, drop_last=True)
        self.label_dict = dict([(0, 'ship-a'), ( 1,'ship-b'), (2,'ship-c'), (3,'fork-a'), (4,'fork-b'), (5,'fork-c')])
        self.save_path = save_path
        self.embedding_vector_length = vector_length
    
    def generate_embeddings(self):
        test_targets = []
        test_imgs = torch.zeros((0,3,64,64), dtype = torch.float32)
        test_embeddings = torch.zeros((0,self.embedding_vector_length),dtype=torch.float32)
        self.model  = self.model.to(self.device)
        
        for x, y in self.dataloader:
            x = x.cuda()
            x = x[0].permute(0,3,2,1)
            embeddings = self.model(x)
            test_targets.extend(y.detach().cpu().tolist())
            test_embeddings= torch.cat((test_embeddings, embeddings.detach().cpu()),0)
            test_imgs = torch.cat((test_imgs, x.detach().cpu()),0)
            
        return test_embeddings, test_targets
     
    def visualize_tsne(self):
        embeddings, targets = self.generate_embeddings()
        tsne = TSNE(2, verbose = 1)
        tsne_proj = tsne.fit_transform(embeddings)
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize = (8,8))
        num_categories = 6
        for lab in range(num_categories):
            indices = [True if lab in i else False for i in targets]
            ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label=self.label_dict[lab])
        ax.legend(fontsize='large', markerscale=2)
        p = os.path.join(self.save_path, "encoder_analysis.png")
        print(f"TSNE saved at:{p}")
        plt.savefig(os.path.join(self.save_path, "encoder_analysis.png"))

        
            

        
        
    

