import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import trange, tqdm
import torch
from train_loop import train, train_for_epochs, validate


classes = ['Apple', 'Baseball', 'Basketball', 'Blueberry', 'Circle', 'Clock', 'Cookie', 'Donut', 'Face', 'Pizza']

USE_GPU = True
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

# data source path
source_path = "data/QD_Data/"

# images and labels
training_img_file = "QD_Training_Data.npy"
validation_img_file = "QD_Validation_Data.npy"
testing_img_file = "QD_Testing_Data.npy"
training_label_file = "QD_Training_Labels.npy"
validation_label_file = "QD_Validation_Labels.npy"
testing_label_file = "QD_Testing_Labels.npy"

# set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

# data loading
def load_data():
    training_img = np.load(f'{source_path}{training_img_file}')
    validation_img = np.load(f'{source_path}{validation_img_file}')
    testing_img = np.load(f'{source_path}{testing_img_file}')

    training_label = np.load(f'{source_path}{training_label_file}')
    validation_label = np.load(f'{source_path}{validation_label_file}')
    testing_label = np.load(f'{source_path}{testing_label_file}')

    print(f'There are {len(training_img)} samples for training, {len(validation_img)} for validation, and {len(testing_img)} for testing')

    return training_img, validation_img, testing_img, training_label, validation_label, testing_label

# dataset
class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img.transpose((2, 0, 1))

        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label)
    
# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 256)

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# dataloaders
def dataloaders(train_img, val_img, test_img, train_lbl, val_lbl, test_lbl):
    train_dataset = Dataset(train_img, train_lbl)
    val_dataset = Dataset(val_img, val_lbl)
    test_dataset = Dataset(test_img, test_lbl)
    
    # batch size
    batch_size = 128

    # instantiating loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, test_dataset

# training
def train_model(model, train_loader, val_loader):
    # optimizer + loss function
    optimizer = torch.optim.Adam(model.parameters(),  lr=0.001) # with learning rate 
    criterion = torch.nn.CrossEntropyLoss()

    NUM_EPOCHS = 5  

    history = train_for_epochs(device, NUM_EPOCHS, model, train_loader, val_loader, optimizer, criterion, binary=False)

    return history, criterion

# plotting

# prediction
def predict(model, data):
    model.eval()
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        # predict
        outputs = model(data).squeeze().cpu()
        # get the prediction by selecting the class with the highest probability
        probs = torch.softmax(outputs, 0)
        _, predicted = torch.max(probs, 0)
        
    return predicted, probs

# test sample visualization
def show_prediction(model, test_dataset, testing_img, idx):
    sample, label = test_dataset[idx]
    
    img = testing_img[idx]
    
    pred, probs = predict(model, sample)

    plt.imshow(img)
    plt.title(f'GT: {classes[label]}, Pred: {classes[pred]}, Prob: {probs[pred] * 100}%')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(classes, probs.numpy() * 100)
    plt.xticks(rotation=45)
    plt.ylabel('Probability (%)')
    plt.title('Class Probability Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    print("Device:",device)

    data = load_data()

    train_img,val_img,test_img,train_lbl,val_lbl,test_lbl = data

    train_loader,val_loader,test_loader,test_dataset = dataloaders(
        train_img,val_img,test_img,
        train_lbl,val_lbl,test_lbl
    )

    model = CNN().to(device)

    history, criterion = train_model(model,train_loader,val_loader)

    test_loss, test_acc = validate(device,model,test_loader,criterion,binary=False)

    print("Test accuracy:",test_acc)

    show_prediction(model,test_dataset,test_img,883)


if __name__ == "__main__":
    main()