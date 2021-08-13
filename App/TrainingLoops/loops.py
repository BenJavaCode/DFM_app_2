from torch.autograd import Variable
from torch import nn
import torch
import time
import copy


def test_loop2(model, dl, dataset_sizes, n_classes, device):

    """
    test_loop2(model, dl, dataset_sizes, n_classes, device)
    Description: Loop for testing.
    Params: model = Model instance.
            dl = Dataloader
            dataset_sizes = Length of dataset.
            n_classes = Number of classes.
            device = The device the tensors and model is run on.
    Latest update: 03-06-2021. Added more comments.
    """

    since = time.time()
    predictions = []
    labelslist = []
    inputslist = []
    predict_prob = []  # For model certainty for each class

    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dl:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels.long())  # Wrapping tensors.

        # forward pass
        with torch.set_grad_enabled(False):  # Will not calculate gradients.
            outputs = model(inputs)  # Evaluate inputs on model.
            _, preds = torch.max(outputs, 1)  # Returns maximum value of all elements in input tensor.
            prob = torch.softmax(outputs, dim=1)  # Rescale elems of the n-dimensional tensor to range [0,1] with sum 1.
            top_p, top_class = prob.topk(n_classes, dim=1)  # Get model certainty for each class
            top_p, top_class = top_p.squeeze(0).tolist(), top_class.squeeze(0).tolist()
            pred_ord = [None] * n_classes
            for i in range(n_classes):
                pred_ord[top_class[i]] = top_p[i]  # Get predictions as percentage for each class

            predict_prob.append(["{:.2f}".format(x) for x in pred_ord])
            predictions.extend(preds.tolist())
            labelslist.extend(labels.tolist())
            inputslist.extend(inputs.tolist())
            loss = nn.CrossEntropyLoss()(outputs, labels)  # Get loss

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
    return predictions, labelslist, inputslist, predict_prob, epoch_acc


def train_model(model, optimizer, dl, dataset_sizes, device, num_epochs=10):

    """
    train_model(model, optimizer, dl, dataset_sizes, device, num_epochs)
    Description: Loop for training model.
    Params: model = Model instance.
            optimizer = Optimization algorithm.
            dl = Dataloader.
            dataset_sizes = Dict of len of each dataset.
            device = The device the tensors and model is run on.
            num_epochs = number of epochs.
    Latest update: 03-06-2021. Added more comments.
    """

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # copy model state dict
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

                inputs = inputs.to(device)  # To device, either CPU or coda enabled GPU.
                labels = labels.to(device)
                inputs, labels = Variable(inputs), Variable(labels.long())

                optimizer.zero_grad()  # Zero gradients from last run

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # (inp,dim) Returns maximum value of all elements in input tensor.
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    # backwards pass
                    if phase == 'train':
                        loss.backward()  # Computes gradients (dloss/dx for every parameter), and accumulates.
                        optimizer.step()  # Update params

                # stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  # For each valuation

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model if better than the hitherto best model.
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
