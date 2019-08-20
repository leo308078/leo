import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import progressbar


# import the model
#--------------------------------------------------------------------------------------------------
#################
##  Code Here  ##
#################
#--------------------------------------------------------------------------------------------------
import sys 
sys.path.insert(1, '../Lab1') 
from model import Net

# settings
#--------------------------------------------------------------------------------------------------
TRAINDATA_DIR = '/data/ocr_dataset'
TESTDATA_DIR = '/data/ocr_dataset'
MODEL_WEIGHTS_PATH = 'ocr_modelweights.pth'
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCH = 3
noGPU = 0
#--------------------------------------------------------------------------------------------------


# global variables declaration
#--------------------------------------------------------------------------------------------------
#################
##  Code Here  ##
#################


device = 'cpu' 
if torch.cuda.is_available(): 
  device = 'cuda'    
  torch.cuda.set_device(noGPU)

label_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
               'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
               'Y', 'Z']
                
widgets = [' ',progressbar.Percentage(), ' ', progressbar.Bar('#'),' ', 
           progressbar.Timer(), ' / ', progressbar.ETA()]
#--------------------------------------------------------------------------------------------------


# datasets
#--------------------------------------------------------------------------------------------------
class ANPR_dataset(Dataset):
    def __init__(self, path_to_csv_file, root_dir):
        self.dataframe = pd.read_csv(path_to_csv_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        path_to_img = os.path.join(self.root_dir, self.dataframe.iloc[idx,0])
        image = Image.open(path_to_img).convert('L')
        transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
        image = transform(image)
        label = label_chars.index(self.dataframe.iloc[idx,1])
        return (image, label)
        
trainset = ANPR_dataset(os.path.join(TRAINDATA_DIR, 'dataset.csv'), TRAINDATA_DIR)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testset = ANPR_dataset(os.path.join(TESTDATA_DIR, 'dataset.csv'), TESTDATA_DIR)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,  shuffle=True, num_workers=NUM_WORKERS)
#--------------------------------------------------------------------------------------------------


# models declaration
#--------------------------------------------------------------------------------------------------
model = Net()
model = model.to(device)
#--------------------------------------------------------------------------------------------------


# training
#--------------------------------------------------------------------------------------------------
def EvalAcc():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        widgets[0] = 'Evaluate: '
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(testloader)).start()
        for i, data in enumerate(testloader, 1):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicts = torch.max(outputs, 1)
            correct += (predicts == labels).sum().item()
            total += BATCH_SIZE
            pbar.update(i)
        print('')
        accuracy = float(correct) / total * 100
        print('Accuracy: {:.1f}%'.format(accuracy))
    model.train()
    return accuracy

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Training starts ...')
print('Total epoches: {:d}\n'.format(EPOCH))
for epoch in range(EPOCH):
    widgets[0] = 'Epoch ' + str(epoch + 1) + ': '
    #################
    ##  Code Here  ##
    #################
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(trainloader)).start() 
    for i, data in enumerate(trainloader, 1):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #################
        ##  Code Here  ##
        #################
        pbar.update(i)

    print('')
    EvalAcc()
    print('')
print('Finish training')
print('----------------------------------------------------------------')
#--------------------------------------------------------------------------------------------------