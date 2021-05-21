from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import time
import copy


# Loop for testing
def test_loop2(model, dl, dataset_sizes, n_classes, device):  # device is the hardware the code will run on
    since = time.time()
    predictions = []
    labelslist = []
    inputslist = []
    predict_prob = []

    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels.long())

        # forward pass
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # inp, dim
            prob = torch.softmax(outputs, dim=1)
            top_p, top_class = prob.topk(n_classes, dim=1)
            top_p, top_class = top_p.squeeze(0).tolist(), top_class.squeeze(0).tolist()
            pred_ord = [None] * n_classes
            for i in range(n_classes):
                pred_ord[top_class[i]] = top_p[i]

            predict_prob.append(["{:.2f}".format(x) for x in pred_ord])
            predictions.extend(preds.tolist())
            labelslist.extend(labels.tolist())
            inputslist.extend(inputs.tolist())
            loss = loss = nn.CrossEntropyLoss()(outputs, labels)

        # stats
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes

    print('\n')

    time_elapsed = time.time() - since
    print('TEST complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(' Test Loss: {:.4f} Test Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return (predictions, labelslist, inputslist, predict_prob, epoch_acc)


# For training model, caches the best model at any given epoch
def train_model(model, optimizer, dl, dataset_sizes, device, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dl[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = Variable(inputs), Variable(labels.long())

                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # inp, dim
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    # backwards pass
                    if phase == 'train':
                        loss.backward()  # calculate gradiens
                        optimizer.step()  # update params

                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load weights of best model
    model.load_state_dict(best_model_wts)
    return model, best_acc
