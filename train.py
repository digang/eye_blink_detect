import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_preprocess import data_loader
from model import Model
import torch.optim as optim

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

if __name__ == '__main__':

    # (2586, 26, 34, 1)
    x_train = np.load('./dataset/x_train.npy').astype(np.float32) 
    # (2586, 1)
    y_train = np.load('./dataset/y_train.npy').astype(np.float32)  
    

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = data_loader.dataset(x_train, y_train, transform=train_transform)

    PATH = 'weights/trained.pth'

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    model = Model()
    model.to('cuda')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 50

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0

        model.train()

        for i, data in enumerate(train_dataloader, 0):
            input_1 = data[0].to('cuda')
            labels = data[1].to('cuda')

            #model 의 input 은 (-1, channel , width , height)
            #npy 의 형태는 (width, height, channel)
            input = input_1.transpose(1, 3).transpose(2, 3)

            optimizer.zero_grad()

            outputs = model(input)
            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

            # 2586 // 32 는 대략 80.xx이다. 따라서 80개의 batch 는 1 epoch 를 상징한다
            if i % 80 == 79:
                print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                    epoch + 1, epochs, running_loss / 80, running_acc / 80))
                running_loss = 0.0

    print("learning finish")
    torch.save(model.state_dict(), PATH)