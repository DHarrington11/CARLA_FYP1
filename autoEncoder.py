import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from torchvision import datasets
import torch.utils.data
import os

def get_weights2(data):
    inv_norms=[]
    for i in range(3000):
        t = np.load("/home/dh26/Documents/Carla/gym-carla/images/AE_Sem/"+dataset[i])
        tag_percentage = [] 
        pixels = [] 
        for i in range(13): 
            im = np.zeros((128,128))
            im[np.where(t==i)[0], np.where(t==i)[1]]=1
            total = im.sum()
            total_percentage = (total/16384)*100
            tag_percentage.append(total_percentage)
            pixels.append(total)
        
        norm =[]
        for x in pixels:
            normalized = (x-min(pixels))/(max(pixels)-min(pixels))
            norm.append(normalized)
        
        inv_norm = []
        for n in norm:
            inv = 1- n
            inv_norm.append(inv)

        inv_norm = inv_norm/sum(inv_norm)
        
        inv_norms.append(inv_norm)
    

    return inv_norms

dataset = [names for names in list(os.listdir("/home/dh26/Documents/Carla/gym-carla/images/AE_Sem"))]
b = get_weights2(dataset)

avg_list= []
for i in range(13):
    class_list = []
    for j in range(len(b)-1):
        x = b[j][i]
        class_list.append(x)
    avg = sum(class_list)/len(class_list)
    avg_list.append(avg)

avg_weights = avg_list
avg_weights = torch.Tensor(avg_weights)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.02), ##Leaky ReLU
            nn.Conv2d(32,64, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Conv2d(64,128,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Conv2d(128,256,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Conv2d(256,512,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Conv2d(512,64,kernel_size=4, stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,512,4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(512,256,4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(256,128,4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(128,64,4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(64,32,4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.02),
            nn.ConvTranspose2d(32,13,4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.loss = nn.CrossEntropyLoss(weight= avg_weights)
        self.learning_rate = 1e-3
        self.optimiser = optim.Adam(self.parameters(), self.learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device='cpu'
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = ["/home/dh26/Documents/Carla/gym-carla/images/AE_Sem/"+names for names in list(os.listdir("/home/dh26/Documents/Carla/gym-carla/images/AE_Sem")[0:-48])]
        
#     def __getitem__(self, index):
#         x = np.load(self.data[index])
#         return x
    
#     def __len__(self):
#         return len(self.data)

# dataset = MyDataset()
# loader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True
# )

# dataset = [names for names in list(os.listdir("/home/dh26/Documents/Carla/gym-carla/images/AE_Sem"))]
# a = np.load("/home/dh26/Documents/Carla/gym-carla/images/AE_Sem/"+dataset[0])
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #device='cpu'

# model = autoencoder()
# # weights = get_weights(a)
# # tensor_transfrom = transforms.ToTensor()
# # weights = torch.Tensor(weights)

# print('Training Started.....')
# # if torch.cuda.is_available():
# #     model = autoencoder.cuda()

# batch_size=32

# dataset = MyDataset()
# loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True
# )

# #optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
# epochs = 10
# outputs = [] 
# losses=[]

# for epoch in range(epochs):
#     print('Continuing...')
#     print("Epoch:{}".format(epoch))
#     running_loss=0.0
#     for i, data in enumerate(loader, 0):

#         inputs = data
#         #print(inputs.shape)
#         #if inputs.shape==[48, 128, 128]:
#         #    print(inputs)
#         inputs = inputs.reshape(batch_size,1,128,128)
#         x = inputs.to(device).float()        

#         model.optimiser.zero_grad()

#         outputs,latent_space = model.forward(x)

#         x=x.reshape(batch_size,128,128)
       
#         # print(type(outputs))
#         # print(type(x))
#         # print(outputs.shape)
#         # print(x.shape)
        
#         loss=model.loss(outputs,x.long())
#         loss.backward()
#         model.optimiser.step()

#         running_loss += loss.item()
#         losses.append(loss)
#         losses.append(sum(losses)/len(losses))
#         if i % 2000 == 1999:
#             print('f[{epoch + i}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss=0.0
# # plt.plot(losses, label = 'Training Loss')
# # plt.show()
# fig, ax = plt.subplots()
# ax.plot(losses, label='Loss', color='blue')
# ax.plot(losses,color='red', lw=4, ls='--', label="average plot")
# plt.legend(loc=0)
# plt.show()
# print('Finished Training')

# path = '/home/dh26/Documents/Carla/gym-carla/AutoEncoder.pth'
# torch.save(model.state_dict(),path)

if __name__ == '__main__':
  autoencoder()