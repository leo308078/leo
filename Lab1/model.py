import torch
from torch import nn


# models definition
#--------------------------------------------------------------------------------------------------
def conv_bn_relu(inc, outc, stride):
    return nn.Sequential(
        nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=outc),
        nn.ReLU6(inplace=True)
    )
    
def convdw_bn_relu(inc, outc, stride):
    return nn.Sequential(
        nn.Conv2d(inc, inc, kernel_size=3, stride=stride, padding=1, groups=inc, bias=False),
        nn.BatchNorm2d(num_features=inc),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=outc),
        nn.ReLU6(inplace=True)
    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #################
        ##  Code Here  ##
        #################
        self.layer1=conv_bn_relu(1,4,2)
        
        self.layer2=convdw_bn_relu(4,4,2)
        
        self.layer3=convdw_bn_relu(4,8,2)
        
        self.layer4=convdw_bn_relu(8,16,2)
        
        self.Dropout=nn.Dropout(0.5)
        
        
        
        self.fc1=nn.Linear(64, 34)
        
        
    def forward(self, x):
        #################
        ##  Code Here  ##
        #################
        
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(-1, 64)
        x=self.Dropout(x)
        x=self.fc1(x)
        return x    
#--------------------------------------------------------------------------------------------------