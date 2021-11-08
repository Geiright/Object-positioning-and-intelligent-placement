import torch
from model import CoordNet
from torch import LongStorage, optim
from torch.utils.data import Dataset
import numpy as np




class coordataset(Dataset):
    def __init__(self, lines=42):
        super().__init__()

        f = open("data.txt", "r")
        
        self.pixels = []
        self.coords = []
        self.lines = lines  #有多少组数据

        for i in range(self.lines):
            x = f.readline()
            
            #print("x=",x)
            x = x.strip('\n')               #去掉换行符
            
            xsp = x.split(',')
            #print("xsp=", xsp)
            self.pixels.append([float(xsp[0]),  float(xsp[1])])
            self.coords.append([float(xsp[2]), float(xsp[3]), float(xsp[4])])
            #print("pixel=", self.pixels)
            #print("coord=", self.coords)

    def __getitem__(self, index):
        #print(self.pixels[index])
        #print(self.coords[index])
        return self.pixels[index], self.coords[index]

    def __len__(self):
        return len(self.pixels)

def coordtrans(pixel, path = 'nice3.pth'):   
    model = CoordNet()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("-----coordinate transformation-----")
    with torch.no_grad():
        input = torch.tensor(pixel)  # input pixel coord (225,292)
        output = model(input)        # output arm coord (155, -190, 70)
    
    return output

if __name__ == '__main__':
    print("---------------init model------------------")
    epochs = 30000
    batchsize = 1
    learning_rate = 0.001
    path = '20211105final.pth'
    pixel = [1085.0,156.0]

    model = CoordNet()

    state_dict = torch.load('coordtrans/nice3.pth')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("---------------infer------------------")
    with torch.no_grad():
        #pixel[0] = pixel[0].type(torch.LongTensor)
        #pixel[1] = pixel[1].type(torch.LongTensor)
        input = torch.tensor(pixel)
        output = model(input)
        print(output)