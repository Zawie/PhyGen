import torch
import torch.nn as nn
import dataHandler
import torch.optim as optim

trainset = dataHandler.SequenceDataset(folder='train')
testset = dataHandler.SequenceDataset(folder='test')

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=True)

class dnn1(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(80, 80, 1, groups=20),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Conv1d(80, 32, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(32, 3)

    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """
        x = x.view(x.size()[0], 80, -1)
        x = self.conv(x).squeeze(dim=2)
        return self.classifier(x)


class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)





def Train(model, epochs=1):
    #define critierion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #training cycle
    print("Training...")
    for epoch in range(epochs):
        print('\tEpoch:',epoch+1)
        for i, data in enumerate(train_loader,0):
            #get the input & labels
            inputs,labels = data
            #inputs = inputs.view(inputs.shape[0],-1)

            #zero gradient
            optimizer.zero_grad()
            
            #forward, backward, optimize
            output = model(inputs)
            #print("Output:",output)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            
            #print statistics
            if (i%10==0):
                print("\t\tLoss:",loss.item())
        print("Done training!")


def Test(model):
    print("Testing...")
    trials = len(testset)
    successes = 0
    for i in range(trials):
        inputs,label = next(iter(test_loader))
        #inputs = inputs.view(inputs.shape[0],-1)
        output = model(inputs).tolist()[0]
        label = label.tolist()[0]
        if label.index(max(label)) == output.index(max(output)):
            successes +=1
    print('Done testing!')
    print("\t"+str(successes/trials*100)+"%",str(successes)+"/"+str(trials))

def Save(model):
    print("Saving model...")
    torch.save(model.state_dict(), "models/alpha")
    print("Model saved!")

#Set model
net = dnn1() 
#Load
net.load_state_dict(torch.load('models/alpha'))
net.eval()
for i in range(25):
    print("CYCLE:",i+1)
    Train(net)
    Test(net)
    Save(net)