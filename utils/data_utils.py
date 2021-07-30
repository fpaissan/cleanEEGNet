from torch.utils.data import Dataset
from torch.utils.data import Subset

from scipy.stats import zscore
from numpy import genfromtxt
from tqdm import tqdm
import numpy as np
import pickle
import os
import utils.params as p
import random


class EEGDataset(Dataset):

    def __init__(self, data_dir):
        self.dir = data_dir
        self.data_files = []
        self.labels_files = []

        for r, d, f in os.walk(data_dir):
            for file in f:
                name_split = file.split("_")
                if name_split[-1] == 'labels.pkl':
                    self.labels_files.append(data_dir + "/" + str(file))
                elif name_split[-1] == 'data.pkl':
                    self.data_files.append(data_dir + "/" + str(file))

        self.data_files = np.sort(self.data_files)
        self.labels_files = np.sort(self.labels_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, i):
        with open(self.data_files[i],"rb") as f:
            data = pickle.load(f)[:, 0:p.dataset_min_length]
        with open(self.labels_files[i],"rb") as f:
            labels = pickle.load(f)

        data = zscore(data)
        data = np.expand_dims(data,0)
        return (data, labels)


def dataset_split(dataset : EEGDataset, ratio: int):
    print("dataset split")
    tot_dataset = len(dataset)
    indices = []
    subjects = []

    # get all subjects identifiers in a given dataset i.e. subjects = ['subject-1', 'subject-2', ... ]
    for f in dataset.data_files:
        file_name = f.replace(dataset.dir+'/','')
        file_name = file_name.split("_")
        sub = file_name[0]
        if not sub in subjects:
            subjects.append(sub)

    # get separate indices lists for each subject from master dataset
    # i.e indeces = [[1,2,3],[4,5,6,7],[8,9] .. ]] each sublist contains the indeces of each subject kept separately 
    for s in subjects:
        sub_indeces = []
        i = 0
        for f in dataset.data_files:
            if s in f:
                sub_indeces.append(i)
            i += 1
        indices.append(sub_indeces)
    
    # subject indeces-list are shuffled subject-wise i.e. indeces = [[4,5,6,7],...,[8,9],...,[1,2,3]]

    random.Random(1729).shuffle(indices)

    partial_count = 0 #accumulates how many indeces are currently in the first subset in each iteration
    composite_ratio = 0 #accumulates the ratio in each iteration
    subset_ids = [] #will contain the indeces of the new subset
    
    while(composite_ratio < ratio):
        partial_count += len(indices[0])
        composite_ratio = partial_count/tot_dataset
        if(composite_ratio < ratio):
            subset_ids.append(indices[0])
            del indices[0] #indeces that are chosen for the first subset are removed from the second

    # flatten lists -> list of list is flatten to a single list 
    subset_ids = [val for sublist in subset_ids for val in sublist]
    indices = [val for sublist in indices for val in sublist]
    
    if (len(indices) == 0):
        raise ValueError("The specified ratio produces one of the subsets to be empty, please change the ratio")
        
    #print the actual ratio obtained avoiding data leakage
    print("Actual ratios: ", len(subset_ids)/tot_dataset, len(indices)/tot_dataset)

    #print("train: ", subset_ids, len(subset_ids))
    #print("test: ", indices, len(indices))
    
    #create subsets from the indeces of the 2 lists using the pytorch function 
    subset_first = Subset(dataset,subset_ids)
    subset_second = Subset(dataset,indices)

    return subset_first, subset_second

