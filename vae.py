from __future__ import print_function, division 
import argparse 
import torch 
# import torch.utils.data 
# from torch import nn, optim 
# from torch.nn import functional as F 
# from torchvision import datasets, transforms 
# from torchvision.utils import save_image 
import torchvision 
import os, sys 
if not os.path.isdir("./results"):
    os.system("mkdir results")
    
parser = argparse.ArgumentParser(description='VAE MNIST') 
parser.add_argument('--batch-size', type=int, default=128, metavar='N', \
    help='input batch size for training (default=128)')
# metavar - A name for the argument in usage messages.
parser.add_argument('--epochs', type=int, default=10, metavar='N',\
    help='number of epochs to train (default=10)')
parser.add_argument('--no-cuda', action='store_true', default=False, \
    help='enables CUDA training, default is with cuda (no-cuda is False)')
parser.add_argument('--seed', type=int, default=1, metavar='S', \
    help='random seed (default=1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', \
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() 
print("Using cuda is {}".format(args.cuda)) 
torch.manual_seed(args.seed) 
cuda0 = torch.device("cuda:0" if args.cuda else "cpu") 
# The torch.device contains a device type ('cpu' or 'cuda') 
# and optional device ordinal for the device type. 
# If the device ordinal is not present, this represents 
# the current device for the device type; 
# e.g. a torch.Tensor constructed with device 'cuda' 
# is equivalent to 'cuda:X' where X is the result of torch.cuda.current_device().
# In [5]: torch.cuda.is_available()                                    
# Out[5]: True
# In [4]: torch.cuda.current_device()                                  
# Out[4]: 0
# In [6]: cuda0 = torch.device("cuda:0")                               
# In [7]: tsor = torch.randn((2,3),device=cuda0)                       
# In [8]: tsor.device()                                                
# ---------------------------------------------------------------------
# TypeError                           Traceback (most recent call last)
# <ipython-input-8-489bc26caada> in <module>
# ----> 1 tsor.device()
# TypeError: 'torch.device' object is not callable
# In [9]: tsor.device                                                  
# Out[9]: device(type='cuda', index=0)
# In [10]: torch.device("cuda:0") == torch.device("cuda", 0)           
# Out[10]: True
kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {} #num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
# kwargs['batch_size'] ... or look below:
kwargs.update({'batch_size':args.batch_size, 'shuffle':True}) # batch_size default=1
# In [11]: def pkwargs(**kwargs):
#     ...:     for a, b in kwargs.items():
#     ...:         print(a,b)
# In [13]: pkwargs(**kwargs)
# ('pin_memory', True)
# ('num_workders', 1)
# https://pytorch.org/docs/stable/data.html 
train_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor()),
    # batch_size=args.batch_size, 
    # shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor()),
    # batch_size=args.batch_size, 
    # shuffle=True,
    **kwargs 
)
class VAE(torch.nn.Module):
    def __init__(self,):
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(784, 400) 
        self.fc21 = torch.nn.Linear(400, 20) 
        self.fc22 = torch.nn.Linear(400, 20) 
        self.fc3 = torch.nn.Linear(20, 400) 
        self.fc4 = torch.nn.Linear(400, 784) 
    def encode(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x)) 
        return self.fc21(h1), self.fc22(h1) 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # logvar.exp() == torch.exp(logvar) 
        eps = torch.randn_like(std) # 'randn_like' Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1. torch.randn_like(input) is equivalent to torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
        
        return eps.mul(std).add_(mu) 
    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z)) 
        return torch.sigmoid(self.fc4(h3)) 
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) 
        z = self.reparameterize(mu, logvar) 
        return self.decode(z), mu, logvar 

model = VAE().to(cuda0) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

def loss_function(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD 
# batch size = 11, here's what I found:
# In [89]: len(train_loader.dataset)                                              
# Out[89]: 60000

# In [90]: len(test_loader.dataset)                                               
# Out[90]: 10000

# In [91]: len(train_loader)                                                      
# Out[91]: 5455

# In [92]: len(test_loader)                                                       
# Out[92]: 910

def train(epoch):
    model.train() 
    train_loss = 0 
# In [60]: train_loader.dataset[0][0].shape                                       
# Out[60]: torch.Size([1, 28, 28])

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(cuda0) 
        optimizer.zero_grad() 
        recon_batch, mu, logvar = model(data) 
        loss = loss_function(recon_batch, data, mu, logvar) 
        loss.backward() 
        train_loss += loss.item() 
        optimizer.step() 
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(\
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item() / len(data)))
                # data is a batch of data, len(data) = batch_size
                # batch_idx is the index of batches not samples
                # e.g. you have 600 samples, 
                # batch size = 10 = len(data) 
                # total batch index from 0 to 60
                # len(train_loader.dataset) == number of total samples
                # len(train_loader) returns the number of batches = len(train_loader.dataset) / train_loader.batch_size 
# In [50]: a = torch.Tensor([12])                                                 
# In [51]: a.item()                                                               
# Out[51]: 12.0
    print('====> Epoch: {} Average loss: {:.4}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval() 
    test_loss = 0 
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(cuda0) 
            recon_batch, mu, logvar = model(data) 
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                torchvision.utils.save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset) 
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch) 
        with torch.no_grad():
            sample = torch.randn(64, 20).to(cuda0) 
            sample = model.decode(sample).cpu() 
            torchvision.utils.save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png') 










