#External modules
import torch
from torch.utils.data import Dataset, DataLoader
import pyvolve
import pickle
import os

#Helper functions
def save(datapoint,id,folder=""):
    """
        Input: datapoint, tuple (X,y)
                id, integer to represent file name/index
        Saves datapoint into a pickle file in the data folder
    """
    pickle.dump(datapoint,open("data/"+folder+"/"+str(id),"wb"))
def load(id,folder=""):
    """
        Id: file name (integer)
    """
    return pickle.load(open("data/"+folder+"/"+str(id),"rb"))

def hotencode(sequence):
    """ 
        Hot encodes inputted sequnce
        "ATGC" -> [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    """
    code_map = {"A":[1,0,0,0],
                "T":[0,1,0,0],
                "G":[0,0,1,0],
                "C":[0,0,0,1]}
    final = []
    for char in sequence:
        final.append(code_map[char])
    return final

#Generate functions
import pyvolve
tree_map = ["alpha","beta","charlie"]

def simulate(tree_index,length):
    """
        Inputs: tree (integer 0-2)
        Outputs: array of 4 sequences, using the tree from above
    """
    tree = tree_map[tree_index]
    my_tree = pyvolve.read_tree(file = "trees/"+tree+".tre")

    #Idk weird pyvolve paramets
    parameters_omega = {"omega": 0.65}
    parameters_alpha_beta = {"beta": 0.65, "alpha": 0.98} # Corresponds to dN/dS = 0.65 / 0.98
    my_model = pyvolve.Model("MG", parameters_alpha_beta)

    # Assign the model to a pyvolve.Partition. The size argument indicates to evolve 250 positions (for a codon alignment, this means 250 codons, i.e. 750 nucleotide sites)
    my_partition = pyvolve.Partition(models = my_model, size = length)

    # Evolve!
    my_evolver = pyvolve.Evolver(partitions = my_partition, tree = my_tree, ratefile = None, infofile = None)
    my_evolver()

    #Extract the sequences
    simulated_sequences = list(my_evolver.get_sequences().values())
    return simulated_sequences

def generatePoint(tree,length = 100):
    sequences = simulate(tree,length)
    #encode sequences
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequences.append(hotencode(sequence))
    #store as datapoint
    X = encoded_sequences
    y = tree
    datapoint = (X,y)
    return datapoint

def generateData(amount=10000,length=100,folder="train"):
    """
        Amount: Amount of trees to generate. Will generate 3 for every 1.
        Trees will be stored in pickle files in /data. (dataset will read from this folder)
    """
    #Generate Alpa, Beta, Charlie trees
    count = 0
    print("Generating data...")
    for i in range(amount):
        if (i%100 == 0):
            print(str(i/amount*100)+"%")
        for tree in [0,1,2]: #0:alpha, 1:beta, 2:charlie
            datapoint = generatePoint(tree,length)
            #store datapoint as pickle file
            save(datapoint,count,folder=folder)
            count += 1
    print("Generation complete!")

#### Format data into pytorch dataset
class SequenceDataset(Dataset):
    def __init__(self,folder):
        #Compute data
        self.folder = folder

    def __getitem__(self,index):
        (sequence,tree_index) = load(index,folder=self.folder)
        label = [0,0,0]
        label[tree_index] = 1
        X = torch.Tensor(sequence)
        y = torch.Tensor(label)
        return X,y
    
    def __len__(self):
        _, _, files = next(os.walk("data/"+self.folder))
        return len(files)

#generateData(amount=10000,length=100,folder='train')
#generateData(amount=1000,length=100,folder='test')