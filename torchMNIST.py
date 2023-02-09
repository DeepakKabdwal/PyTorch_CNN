import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
import torch.optim as op
import torch.nn.functional as F

PATH = './trainedModel/mnist_cnn_net.pth'
device  = 'cpu'

transform = trans.Compose([trans.ToTensor(),
                           trans.Normalize((0.5, ), (0.5, ))])

#adding datasets
trainset = torchvision.datasets.MNIST('mnist',
                                      train = True,
                                      download = True,
                                      transform = transform)
testset = torchvision.datasets.MNIST('mnist',
                                     train = False,
                                     download = True,
                                     transform = transform)

#adding dataloaders

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = 128,
                                          shuffle = True,
                                          num_workers = 0)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size = 128,
                                         shuffle = False,
                                         num_workers = 0)

dataiter = iter(trainloader)
images, labels = next(dataiter)

#model training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
net.to(device)
#print(net)

criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

#training model

epochs = 10
epoch_log = []
loss_log = []
accuracy_log = []

for epoch in range (epochs):
    print(f'Starting Epoch: {epoch + 1}...')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i % 50 == 49:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, dim = 1)
                    total+=labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct/total
                epoch_num = epoch+1
                actual_loss = running_loss / 50
                print(f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss=0.0
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

torch.save(net.state_dict(), PATH)
