import torch.nn as nn
import pdb
import torch
from torch.autograd import Variable
import pdb
import numpy as np
pdb.set_trace()
loss_function = nn.CrossEntropyLoss()
input = torch.Tensor(np.zeros((3,20)))
input[0,3] = 1
input[1,8] = 1
input[2,16] = 1
#input = torch.Tensor([[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
#input = torch.Tensor([[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0]])
#input = torch.Tensor([[0,0,1,0],[1,0,0,0],[0,1,0,0]])
#target = torch.LongTensor([2,6,7])
#target = torch.LongTensor([2,6,7])
#target = torch.LongTensor([3,8,16])
#target = torch.LongTensor([0])
#loss = loss_function(Variable(input), Variable(target))
#print(loss)

## loss = 1.4612 10分类
## loss = 0.7437 4分类

import random
import pdb
pdb.set_trace()
a = [1,2,3,4,5,6,7,8,9,10]
random.seed(10)
e = random.random()
random.seed(10)
h = random.random()
b = random.choice(a)
random.seed(10)
c = random.choice(a)
random.seed(20)
f = random.random()
random.seed(10)
g = random.random()
d = random.choice(a)
