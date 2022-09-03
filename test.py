import torch
import torch.nn as nn
import load_dataset
import torch
from main import ConvNCF
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# model = Model(*args, **kargs)
#model = ConvNCF(*args, **kargs)
model = torch.load('model.pth')
model.eval()
for param in model.parameters():
    print(param)
