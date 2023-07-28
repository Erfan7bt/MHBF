import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy.linalg as la
from plotter import visCov, visITO,visUA,visWP

from data import *
from models import *


# Below snippit taken from PyTorch quickstart tutorial
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(device)

pause_lengths = np.arange(25, 51, 5)
model_rank2= RNN(rank=2)

print("Rank 2 training")
for pl in pause_lengths:
    print("Pause length: ", pl)
    p_data = ParametricDataset(32000, pause=pl)
    train_dataloader = DataLoader(p_data, batch_size=32)
    
    if pl == pause_lengths[-1]:
        train(train_dataloader, model_rank2, device,T=5,plotloss=True)
    else:
        train(train_dataloader, model_rank2, device,T=5,plotloss=False)
#save model
torch.save(model_rank2, "model_rank2.pt")


print("Rank 1 training")
model_rank1 = RNN(rank=1)

s_data = SingleDecisionDataset(32000)
train_dataloader = DataLoader(s_data, batch_size=32)
train(train_dataloader, model_rank1, device,plotloss=True)
#save model
torch.save(model_rank1, "model_rank1.pt")

# visITO(SingleDecisionDataset, model_rank1)

# visITO(ParametricDataset, model_rank2)