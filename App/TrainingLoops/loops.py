from torch.autograd import Variable
from torch import nn
import torch
import time
import copy
from labml_helpers.module import Module


class SquaredErrorBayesRisk(Module):
    """
    <a id="SquaredErrorBayesRisk"></a>
    ## Bayes Risk with Squared Error Loss
    Here the cost function is squared error,
    $$\sum_{k=1}^K (y_k - p_k)^2 = \Vert \mathbf{y} - \mathbf{p} \Vert_2^2$$
    We integrate this cost over all $\mathbf{p}$
    \begin{align}
    \mathcal{L}(\Theta)
    &= -\log \Bigg(
     \int
      \Big[ \sum_{k=1}^K (y_k - p_k)^2 \Big]
      \frac{1}{B(\color{orange}{\mathbf{\alpha}})}
      \prod_{k=1}^K p_k^{\color{orange}{\alpha_k} - 1}
     d\mathbf{p}
     \Bigg ) \\
    &= \sum_{k=1}^K \mathbb{E} \Big[ y_k^2 -2 y_k p_k + p_k^2 \Big] \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big)
    \end{align}
    Where $$\mathbb{E}[p_k] = \hat{p}_k = \frac{\color{orange}{\alpha_k}}{S}$$
    is the expected probability when sampled from the Dirichlet distribution
    and $$\mathbb{E}[p_k^2] = \mathbb{E}[p_k]^2 + \text{Var}(p_k)$$
     where
    $$\text{Var}(p_k) = \frac{\color{orange}{\alpha_k}(S - \color{orange}{\alpha_k})}{S^2 (S + 1)}
    = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$$
     is the variance.
    This gives,
    \begin{align}
    \mathcal{L}(\Theta)
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big) \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] +  \mathbb{E}[p_k]^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( \big( y_k -\mathbb{E}[p_k] \big)^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( ( y_k -\hat{p}_k)^2 + \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1} \Big)
    \end{align}
    This first part of the equation $\big(y_k -\mathbb{E}[p_k]\big)^2$ is the error term and
    the second part is the variance.
    """

    def forward(self, evidence: torch.Tensor, target: torch.Tensor, dirichlet_threshold=0.01):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # $\color{orange}{\alpha_k} = e_k + 1$
        alpha = (evidence) + dirichlet_threshold
        # $S = \sum_{k=1}^K \color{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)
        # $\hat{p}_k = \frac{\color{orange}{\alpha_k}}{S}$
        p = (alpha / strength[:, None]).to(DEVICE)

        # p = evidence / strength[:, None]

        # Convert to one-hot encoded format, with num_classes as vector size
        target = (torch.eye(evidence[0].shape[0])[target]).to(DEVICE)

        # Error $(y_k -\hat{p}_k)^2$
        err = (target - p) ** 2
        # Variance $\text{Var}(p_k) = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$
        var = p * (1 - p) / (strength[:, None] + 1)

        # Sum of them
        loss = (err + var).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


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

        for phase in list(dl.keys()):
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
            try:
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            except Exception as e:
                print(e)

        print('\n')
    if len(dl) == 1:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        print('This acc is a placeholder and means nothing')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load weights of best model
    model.load_state_dict(best_model_wts)
    return model, best_acc


def train_model_edl(model, optimizer, dl, dataset_sizes, device, num_epochs=10):

    """
    train_model_edl(model, optimizer, dl, dataset_sizes, device, num_epochs)
    Description: Loop for training model.
    Params: model = Model instance.
            optimizer = Optimization algorithm.
            dl = Dataloader.
            dataset_sizes = Dict of len of each dataset.
            device = The device the tensors and model is run on.
            num_epochs = number of epochs.
    Latest update: 19-10-2021.
    """

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # copy model state dict
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in list(dl.keys()):
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

                    criterion = SquaredErrorBayesRisk()
                    loss = criterion(outputs, labels, dirichlet_threshold=0.01)

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
    if len(dl) == 1:
        best_model_wts = copy.deepcopy(model.state_dict())
        print('This acc is a placeholder and means nothing')
        best_acc = 0.0
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load weights of best model
    model.load_state_dict(best_model_wts)
    return model, best_acc


def test_model_edl(model, dl_test, device, back_idx, dirichlet_threshold=0.01):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    confidenses = []
    data_acum = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dl_test):
            data, target = data.to(device), target.to(device)

            evidence = model(data)
            alphas = [x + dirichlet_threshold for x in evidence]
            Ss = [x.sum() for x in alphas]
            _, pred = torch.max(evidence.data, 1)
            # print(alpha.tolist())

            alphais = [a.tolist()[int(p)] for a, p in zip(alphas, pred.tolist())]

            # alpha_tilde = alphai/S

            # Confidence = 1-beta.cdf(0.5,alphai,S-alphai)
            confids = [(ai/s).item() for ai, s in zip(alphais, Ss)]

            predictions.extend(pred.tolist())
            confidenses.extend(confids)
            targets.extend(target.tolist())
            data_acum.extend(data.tolist())

            total += target.size(0)

            if device == 'cuda':
                correct += pred.eq(target.data).cpu().sum().item()
            else:
                correct += pred.eq(target.data).sum().item()

    """
    accuracy = correct / total
    #pred_confs = [[pred, conf] for pred,conf in zip(predictions, confidenses)]
    #low_conf_corrects = [[pred, conf] for pred, conf, target in zip(predictions, confidenses, targets) if pred == target and conf < 0.5]
    wrongs = [[pred, target, conf] for pred, target, conf in zip(predictions, targets, confidenses) if pred != target and (target == 0 or pred == 0)]
    wrong_thr = 0.98
    wrongs_and_wrongs_by_threshold = len([1 for pred, target, conf in zip(predictions, targets, confidenses) 
                                      if pred != target and conf > wrong_thr and (target == 0 or pred == 0)])
    wrongs_by_threshold_but_right = len([1 for pred, target, conf in zip(predictions, targets, confidenses) 
                                      if pred == target and conf < wrong_thr and target == 0])

    confident_preds = [x for x,c,p in zip(data_acum, confidenses, predictions) if c > wrong_thr and p != 0]
    """
    wrong_thr = 0.98
    labels = [0 if c > wrong_thr and p == back_idx else 1 if c > wrong_thr and p != back_idx
             else 2 if p == back_idx else 3 for c, p in zip(confidenses, predictions)]

    # print(f'low confidence corrects prediction = {low_conf_corrects}')
    # print(f'wrong predictions: [pred, target, conf] =  {wrongs}')
    # print(f'Correctes that would be removed if thr={wrong_thr} : {wrongs_by_threshold_but_right}')
    # print(f'Wrongs that would go through if thr={wrong_thr} : {wrongs_and_wrongs_by_threshold}')
    # print(f'Total number of wrongs = {len(wrongs)}')

    # print(f'len of test dataset = {len(targets)}')
    # print(f'{len(wrongs)/len(targets)} % where wrong guesses')
    # print(f'test accuracy={accuracy}')
    return labels, confidenses


