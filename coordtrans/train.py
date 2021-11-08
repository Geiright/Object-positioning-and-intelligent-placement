import torch
from model import CoordNet
from torch import optim
from torch.utils.data import Dataset, DataLoader





class coordataset(Dataset):
    def __init__(self, lines=48):
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



if __name__ == '__main__':
    print("---------------init model------------------")
    epochs = 10000
    datanum = 42
    batchsize = 1
    learning_rate = 0.001
    lr_place = [1500, 5000]
    path = '20211105final.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoordNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_place, gamma=0.1, last_epoch=-1) 
    criteon = torch.nn.MSELoss()

    print("------------preparing data-----------------")
    '''pixel = []                              #pixel左边为序号，右边为像素坐标、机械臂坐标。
    pixel[0] = [[335,225], [178.4, -196.5, 69.6]]
    pixel[1] = [[513,468], [292.7, -97.4, 70.1]]
    pixel[2] = [[755,408], [257, 22.3, 72]]
    pixel[3] = [[696,290], [198.4, -10.3, 70.3]]
    pixel[4] = [[576,347], [231.7, -69.3, 71.8]]
    pixel[5] = [[455,346], [235.7, -130.2, 69.6]]'''
    
    train_loader = torch.utils.data.DataLoader(
                    coordataset(lines=datanum),batch_size=batchsize, shuffle=False)
    data2 = []
    flag = 0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = float(data[i])
            data = torch.tensor(data).to(device)
            #print("data=", data)

            for i in range(len(target)):
                target[i] = float(target[i])
            target = torch.tensor(target).to(device)
            #print("target=", target)
            output = model(data)
            #print(output)
            #print(target)
            loss = criteon(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            if batch_idx % 100 == 0 and epoch % 10 == 0:
                print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(train_loader), len(train_loader),
                    100.*batch_idx/len(train_loader), loss.item()
                ))
                if loss.item() < 1:
                    flag = flag + 1
                

        if flag >= 5:
            torch.save(model.state_dict(), str(epoch)+'.pth')
            print("nice model here")
            flag = 0

        if epoch >= 2500 and epoch % 500 == 0:
            torch.save(model.state_dict(), str(epoch)+'.pth')
        if epoch + 1 == epochs:
            torch.save(model.state_dict(), path)
