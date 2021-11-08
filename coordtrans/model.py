import torch
import torch.nn as nn

class CoordNet(nn.Module):                          #输入像素坐标，输出机械臂坐标
    def __init__(self):
        super(CoordNet, self).__init__()
    
        self.seq = nn.Sequential(
                        nn.Linear(2, 16),
                        #nn.ReLU(),
                        #nn.LeakyReLU(),
                        nn.Linear(16,16),
                        #nn.ReLU(),
                        nn.LeakyReLU(),
                        #nn.Softplus(),
                        nn.Linear(16, 3)
                        #nn.LeakyReLU()
                        )
    def forward(self, x):
        x = self.seq(x)
        return x 