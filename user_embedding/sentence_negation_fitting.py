from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
from pprint import pprint
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

## change test to train later
entailment_train = torch.load('user_embedding/datasets/entailment_train.pkl').cuda()
negative_train = torch.load('user_embedding/datasets/negative_train.pkl').cuda()
x_train = torch.cat((entailment_train, negative_train), 0)
y_train = torch.cat((negative_train, entailment_train),0)

entailment_train = torch.load('user_embedding/datasets/entailment_test.pkl').cuda()
negative_train = torch.load('user_embedding/datasets/negative_test.pkl').cuda()
x_test = torch.cat((entailment_train, negative_train), 0)
y_test = torch.cat((negative_train, entailment_train),0)

data_train = torch.utils.data.TensorDataset(x_train,y_train)
data_test = torch.utils.data.TensorDataset(x_test,y_test)


training_loader = torch.utils.data.DataLoader(data_train, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=False)

inputDim = x_train.size(1)
outputDim = y_train.size(1)
learningRate = 10.0   ## A high learning rate is used on purpose as the model is simply a constrained linear regression
epochs = 16

model = linearRegression(inputDim, outputDim)
if torch.cuda.is_available():
    model.cuda()

loss_fn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter('user_embedding/runs/negation_fitting_{}'.format(timestamp))


best_vloss = 1000000.0
for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)

    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, des_outputs = data
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = loss_fn(outputs, des_outputs)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            last_loss = running_loss / 500 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(last_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    tb_writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : last_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    tb_writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'user_embedding/models/negation_mat_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)