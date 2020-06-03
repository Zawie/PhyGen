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
    simulated_sequences = my_evolver.get_sequences()
    print(simulated_sequences)
    return simulated_sequences.items()