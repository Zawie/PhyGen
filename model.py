import torch
import torch.nn as nn
import dataHandler
import torch.optim as optim

trainset = dataHandler.SequenceDataset(folder='train')
testset = dataHandler.SequenceDataset(folder='test')

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=5000, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=True)

class PhyNet(nn.Module):
    def __init__(self, input_size=4800): #sequence_length*4 (each vector is 4 long)
        super(PhyNet,self).__init__()
        self.fc1 = nn.Linear(input_size,1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 250)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(250, 100)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

model = PhyNet()
print(model)

#define critierion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training cycle
print("Training...")
for epoch in range(3):
    print('Epoch:',epoch+1)
    for i, data in enumerate(train_loader,0):
        #get the input & labels
        inputs,labels = data
        inputs = inputs.view(inputs.shape[0],-1)

        #zero gradient
        optimizer.zero_grad()
        
        #forward, backward, optimize
        output = model(inputs)
        #print("Output:",output)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        #print statistics
        print("Loss:",loss.item())

print("Done training!")

#Test
print("Testing...")
trials = len(testset)
successes = 0
for i in range(trials):
    inputs,label = next(iter(test_loader))
    inputs = inputs.view(inputs.shape[0],-1)
    output = model(inputs).tolist()[0]
    label = label.tolist()[0]
    if label.index(max(label)) == output.index(max(output)):
        successes +=1

print('Done testing!')
print(str(successes/trials*100)+"%",str(successes)+"/"+str(trials))
