import pathlib
import pickle

from scapy.all import *  # noqa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter

from VPN_Dataset import MTU_LENGTH

dirname = pathlib.Path.cwd()
plt.set_loglevel('info')

# Set the default dtype and device
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(torch.device('cuda'))
    print("using cuda:", torch.cuda.get_device_name(0))
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(torch.device('cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PcapDataset(Dataset):

    def __init__(self, normal_abnormal=False, test=False) -> None:
        super().__init__()

        self.normal_abnormal = normal_abnormal
        self.test = test

        with open("data/packets.pickle", 'rb') as file:
            data = file.read()
            self.packets = pickle.loads(data)

    def __len__(self):
        return len(self.packets)

    def get_type_count(self):
        if self.normal_abnormal:
            return 2
        raise Exception("unknown classify type")

    def __getitem__(self, index):
        if index >= len(self.packets):
            raise StopIteration

        row = self.packets[index]

        normal_abnormal, content = row

        target = torch.zeros(self.get_type_count())

        if self.normal_abnormal:
            label = 0 if normal_abnormal == 'normal' else 1

        target[label] = 1.0

        if len(content) < MTU_LENGTH:
            content += b'\0' * (MTU_LENGTH - len(content))

        content = content[:MTU_LENGTH]

        assert (len(content) == MTU_LENGTH)

        data = np.frombuffer(content, dtype=np.uint8, count=MTU_LENGTH)

        # Specify data type
        image = torch.tensor(data.copy(), dtype=torch.float32) / 255.0
        image = image.view(1, MTU_LENGTH)

        return label, image, target


class Classifier(nn.Module):

    def __init__(self, type_count):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 128, kernel_size=5, stride=2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(128, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Flatten(),

            nn.LazyLinear(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.LazyLinear(type_count),
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self.right = 0
        self.total = 0

    def forward(self, inputs):
        return self.model(inputs)

    def train_step(self, label, inputs, targets):
        self.model.train()
        inputs = inputs.view(-1, 1, MTU_LENGTH)
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        predicted = torch.max(outputs.data, 1)[1]

        self.right += (predicted == label).sum().item()
        self.total += len(label)

        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total = 0
        right = 0
        with torch.no_grad():
            for label, image, target in dataloader:
                image = image.view(-1, 1, MTU_LENGTH)
                outputs = self.forward(image)
                predicted = torch.max(outputs.data, 1)[1]
                total += label.size(0)
                right += (predicted == label).sum().item()
        return right / total

# Modify the kwargs parameter
kwargs = {
    'normal_abnormal': True,
}

epoch = 2
test_percent = 0.2

writer = SummaryWriter()

# Reload the dataset and create the classifier
dataset = PcapDataset(**kwargs)
c = Classifier(dataset.get_type_count())

total_count = len(dataset)
test_count = int(total_count * test_percent)
train_count = total_count - test_count

label, image, target = dataset[0]
print(image.device, target.device)

output = c.forward(image.view(1, 1, MTU_LENGTH))
print(output.shape)

trainset, testset = random_split(
    dataset,
    [train_count, test_count],
    torch.Generator(device=device))

train_loader = DataLoader(
    dataset=trainset,
    batch_size=512,
    shuffle=True,
    generator=torch.Generator(device=device),
    drop_last=True,
)

test_loader = DataLoader(
    dataset=testset,
    batch_size=512,
    shuffle=False,
    generator=torch.Generator(device=device),
    drop_last=False,
)

best_acc = 0
early_stop_counter = 0
early_stop_patience = 10

for var in range(epoch):

    tq = tqdm(train_loader)

    c.total = 0
    c.right = 0
    halfway_point = len(train_loader) // 2  # Calculate the position of the halves
    for i, (label, image, target) in enumerate(tq):
        loss = c.train_step(label, image, target)
        acc = c.right / c.total

        writer.add_scalar("loss", loss)
        writer.add_scalar("acc", acc)

        tq.set_postfix(epoch=f"{var}", acc='%.6f' % acc)

        if i == halfway_point:
            print(f"Halfway Accuracy after {halfway_point} iterations: {acc:.6f}")

    tq.close()

    # Evaluate on the test set
    test_acc = c.evaluate(test_loader)
    writer.add_scalar("test_acc", test_acc, var)

    print(f"Epoch {var} - Test Accuracy: {test_acc}")

    # Early stopping check
    if test_acc > best_acc:
        best_acc = test_acc
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered")
        break

# Evaluation of final test data
total = 0
right = 0
tq = tqdm(test_loader)

rights = np.zeros(dataset.get_type_count(), dtype=np.float32)
totals = np.zeros(dataset.get_type_count(), dtype=np.float32)

for label, image, target in tq:
    image = image.view(-1, 1, MTU_LENGTH)
    outputs = c.forward(image)
    predicted = outputs.argmax(dim=1)

    # Number of correct forecasts updated
    correct_predictions = (predicted == label).sum().item()
    right += correct_predictions

    # Update total sample size
    batch_size = label.size(0)
    total += batch_size

    # Update the number of correct predictions and totals for each category
    for i in range(batch_size):
        totals[label[i].item()] += 1
        if predicted[i].item() == label[i].item():
            rights[label[i].item()] += 1

    writer.add_scalar("tacc", right / total)
    tq.set_postfix(acc='%.6f' % (right / total))

tq.close()

print(right, total, right / total)

print(totals)
print(rights)
print(rights / totals)
