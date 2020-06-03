#External modules
import torch
from torch.utils.data import Dataset, DataLoader
import pyvolve
import pickle
import pyvolve

#Helper functions
def save(datapoint,id):
    """
        Input: datapoint, tuple (X,y)
                id, integer to represent file name/index
        Saves datapoint into a pickle file in the data folder
    """
    pickle.dump(datapoint,open("data/"+str(id),"wb"))
def load(id):
    """
        Id: file name (integer)
    """
    return pickle.load(open("data/"+str(id),"rb"))

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

#Generate functions
import pyvolve
tree_map = ["alpha","beta","charlie"]

def generate(tree_index):
    """
        Inputs: tree (integer 0-2)
        Outputs: array of 4 sequences, using the tree from above
    """
    tree = tree_map[tree_index]
    my_tree = pyvolve.read_tree(file = "trees/alpha.tre")

    #Idk weird pyvolve paramets
    parameters_omega = {"omega": 0.65}
    parameters_alpha_beta = {"beta": 0.65, "alpha": 0.98} # Corresponds to dN/dS = 0.65 / 0.98
    my_model = pyvolve.Model("MG", parameters_alpha_beta)

    # Assign the model to a pyvolve.Partition. The size argument indicates to evolve 250 positions (for a codon alignment, this means 250 codons, i.e. 750 nucleotide sites)
    my_partition = pyvolve.Partition(models = my_model, size = 5)

    # Evolve!
    my_evolver = pyvolve.Evolver(partitions = my_partition, tree = my_tree, ratefile = None, infofile = None)
    my_evolver()

    #Extract the sequences
    simulated_sequences = list(my_evolver.get_sequences().values())
    return simulated_sequences

def generateSequences(amount=100):
    """
        Amount: Amount of trees to generate. Will generate 3 for every 1.
        Trees will be stored in pickle files in /data. (dataset will read from this folder)
    """
    #Generate Alpa, Beta, Charlie trees
    count = 0
    for _ in range(amount):
        for tree in [0,1,2]: #0:alpha, 1:beta, 2:charlie
            sequences = generate(tree)
            #encode sequences
            encoded_sequences = []
            for sequence in sequences:
                encoded_sequences.append(hotencode(sequence))
            #store as datapoint
            X = encoded_sequences
            y = tree
            datapoint = (X,y)
            #store datapoint as pickle file
            save(datapoint,count)
            count += 1

generateSequences(amount=1)
print("generation complete")
print(load(0))
print(load(1))
print(load(2))

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
