import torch
from torch import nn

# defin a Net class
class Net(nn.Module): # Your models should also subclass this class.
    def __init__(self, in_c=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=3, stride=1, padding=1) # set padding=1 to keep the spatial size
        
        self.relu1 = nn.ReLU() # relu activation
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # max pooling
        
    def forward(self, x):
        x = self.conv1(x) #[1, 3, 224, 224] -> [1, 32, 224, 224]
        
        x = self.relu1(x) # [1, 3, 224, 224]
        
        x = self.maxpool2d(x)  # [1, 32, 224, 224] -> [1, 32, 112, 112]
        
        return x
    

if __name__ == '__main__':
    net = Net(in_c=3)
    input_data = torch.randn((1, 3, 224, 224))
    
    output = net(input_data)
   
        