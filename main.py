#External modules
import torch
from torch.utils.data import Dataset, DataLoader
import pyvolve
import pickle
#Internal modules
#import simulator

def save(datapoint,id):
    """
        Input: datapoint, tuple (X,y)
                id, integer to represent file name/index
        Saves datapoint into a pickle file in the data folder
    """
    pickle.dump(datapoint,open("data/"+id,"wb"))
def load(id):
    """
        Id: file name (integer)
    """
    pickle.load(open("data/"+id,"rb"))

def hotencode(sequence):
    """ 
        Hot encodes inputted sequnce
        "GATC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {"G":[1,0,0,0],
                "A":[0,1,0,0],
                "T":[0,0,1,0],
                "C":[0,0,0,1]}
    final = []
    for char in sequence:
        final.append(code_map[char])
    return final
        
def generateSequences(amount=100):
    """
        Amount: Amount of trees to generate. Will generate 3 for every 1.
        Trees will be stored in pickle files in /data. (dataset will read from this folder)
    """
    #Generate Alpa, Beta, Charlie trees
    count = 0
    for tree in [0,1,2]: #0:alpha, 1:beta, 2:charlie
        sequences = simulator.generate(tree=tree)
        for sequence in sequences:
            X = hotencode(sequence)
            y = tree
            datapoint = (X,y)
            #store datapoint as pickle file
            save(datapoint,count)
            count += 1

#### Format data into pytorch dataset
class SequenceDataset(Dataset):
    def __init__(self,amount,length):
        #Compute data
        self.x_data = []
        self.y_data = []

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)