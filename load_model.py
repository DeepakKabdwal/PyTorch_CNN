import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as trans
import torch.nn.functional as F


device = 'cuda'
PATH = './trainedModel/mnist_cnn_net.pth'
transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5,), (0.5,))])


# we will load the model trained in the program torchMNIST.py
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


# make a new instance of the model to load it in
net = Net()
net.to(device)

testset = torchvision.datasets.MNIST('mnist',
                                     train=False,
                                     download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         num_workers=0)


net.load_state_dict(torch.load(PATH))

#declaring the iterator for test-sets and feeding them into the previously trained model
testIter = iter(testloader)
images, labels = testIter.next()
outputs = net(images)
_, predicted = torch.max(outputs, 1)
#printing the predictions
print('predicted:   ', ''.join('%ls' % predicted[j].cpu().numpy() for j in range(128)))

