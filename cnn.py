from __future__ import print_function, division 
import argparse 
import torch, torchvision 

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1) 
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1) 
        self.fc1 = torch.nn.Linear(4*4*50, 500) 
        self.fc2 = torch.nn.Linear(500, 10) 
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x)) 
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x)) 
        x = torch.nn.functional.max_pool2d(x, 2, 2) 
        x = x.view(-1, 4*4*50) 
        x = torch.nn.functional.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return torch.nn.functional.log_softmax(x, dim=1) # what's this dim for?


def train(args, model, device, train_loader, optimizer, epoch):
    model.train() 
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad() 
        output = model(data) 
        loss = torch.nn.functional.nll_loss(output, target) 
        train_loss += loss.item()
        # dim issue here.
        loss.backward() 
        optimizer.step() 
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),100.0 * batch_idx / len(train_loader), loss.item()))
            

    print("Epoch {}, average loss = {}".format(epoch, train_loss / len(train_loader.dataset)))

# model.eval() will notify all your layers that you are in eval mode, 
# that way, batchnorm or dropout layers will work in eval model instead of training mode.
# torch.no_grad() impacts the autograd engine and deactivate it. 
# It will reduce memory usage and speed up computations 
# but you won’t be able to backprop (which you don’t want in an eval script).
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) 
            loss = torch.nn.functional.nll_loss(output, target) 
            test_loss += loss.item() 
            pred = output.argmax(dim=1, keepdim=True) 
            print("target.shape = ", target.shape)
            print("pred.shape = ",pred.shape) 
            correct += pred.eq(target.view_as(pred)).sum().item() 
    print("Test set average loss = {:.4f}, accuracy = {}/{} ({:.0f}%)\n".format(
            test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset) 
        ))


def main():
    parser = argparse.ArgumentParser(description='pytorch mnist cnn') 
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',)
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',)
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',)
    parser.add_argument('--save-model', action='store_true', default=True) 
    args = parser.parse_args() 
    args.cuda = not args.no_cuda and torch.cuda.is_available() 
    torch.manual_seed(args.seed) 
    device = torch.device('cuda' if args.cuda else 'cpu') 
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {} 
    kwargs.update({'batch_size':args.batch_size, 'shuffle':True})
    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        **kwargs 
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307), (0.3081,))
            ])
        ),
        **kwargs 
    )
    model = Net().to(device) 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) 

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch) 
        test(args, model, device, test_loader) 
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt") 

if __name__ == "__main__":
    main() 



