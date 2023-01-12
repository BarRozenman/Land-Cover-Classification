import matplotlib
import sklearn.metrics
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.utils
from seaborn import heatmap
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.nn.functional import softmax, one_hot

matplotlib.use('TkAgg')
writer = SummaryWriter('runs/mnt')
class mynn(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(28*20,5,3)
        self.net = nn.Sequential(
            nn.Linear(28*28,10),
            nn.GELU(),
            nn.Linear(10, 10),
        )
    def forward(self,x):
        x = nn.Flatten()(x)
        return self.net(x)

metadata = pd.read_csv('data/geo/metadata.csv')
class_dict = pd.read_csv('data/geo/class_dict.csv')
cv2.imread('data/train/00000001_000.png')
datasets.MNIST
from torchvision.transforms import ToTensor
training_dataset = datasets.FashionMNIST('data',True,download=True,transform=ToTensor())
testing_dataset = datasets.FashionMNIST('data',False,download=True,transform=ToTensor())
plt.imshow(training_dataset.data[0,:,:])

img_grid = torchvision.utils.make_grid([training_dataset.data[0,:,:],training_dataset.data[1,:,:]])
writer.add_image('mnst',img_grid)
plt.imshow(training_dataset.data[0,:,:])
# for count,(X,y) in enumerate(DataLoader(testing_dataset,5)):
#     print(j)

a=1

dl=DataLoader(training_dataset,20)
net = mynn()
loss_fn= nn.CrossEntropyLoss()
opt=Adam([i for i in net.parameters()],lr=0.00001)
size = len(dl)
nn.CrossEntropyLoss()(torch.Tensor([[0,3]]),torch.Tensor([1]).type(torch.long))
examples = iter(testing_dataset)
example_data, example_targets = next(examples)
writer.add_graph(net,example_data.reshape(-1,28*28))#,np.flatten(training_dataset.data[0,:,:]))
running_loss=0.0

yy=[]
res=[]
for batch, (X, y) in enumerate(dl):
    res = net(X)
    opt.zero_grad()
    loss = loss_fn(res, y)
    loss.backward()
    opt.step()
    running_loss +=loss
    writer.add_scalar('train',running_loss,batch+1)
    ress.append(softmax(res))
    yy.append(y)
    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        running_loss= 0
dl=DataLoader(testing_dataset,30)
# writer.add_pr_curve('sdf',torch.cat(yy),torch.cat(ress))
writer.add_pr_curve('sdf',one_hot(torch.cat(yy)) ,torch.cat(ress))
for batch, (X, y) in enumerate(dl):
    res = net(X)
    opt.zero_grad()
    loss = loss_fn(res, y)
    loss.backward()
    opt.step()
    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
_,a=torch.max(one_hot(torch.cat(yy)) , 1)
_,b=torch.max(torch.cat(ress), 1)
aa=sklearn.metrics.confusion_matrix(a,b)
writer.add_figure("Confusion matrix", heatmap(aa/len(a), annot=True).get_figure(), 5)
sklearn.metrics.confusion_matrix(a.detach().numpy(),b.detach().numpy())
writer.close()
