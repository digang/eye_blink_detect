from sklearn import datasets
import torch
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader



#custom library
from model import Model
from data_preprocess import data_loader



def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

PATH = './weights/trained.pth'

x_test = np.load('./dataset/x_val.npy').astype(np.float32)
y_test = np.load('./dataset/y_val.npy').astype(np.float32)

test_transform = transforms.Compose([
    transforms.ToTensor()
    ])
test_dataset = data_loader.dataset(x_test,y_test,transform=test_transform)
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False,num_workers=0)

model = Model()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
count = 0

with torch.no_grad():
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0], test_data[1]

        data = data.transpose(1, 3).transpose(2, 3)

        outputs = model(data)

        acc = accuracy(outputs, labels)
        total_acc += acc

        count = i

    print('avarage acc: %.5f' % (total_acc/count),'%')

print('test finish!')