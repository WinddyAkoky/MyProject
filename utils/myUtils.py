import os
import torch
import torchvision
import torchvision.transforms as transform
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


def test_accuracy():
    # 测试模型的准确率
    pass


def my_imshow(img, NORMALIZE):
    # 画图
    if NORMALIZE:
        img = img * 0.3081 + 0.1307
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


def my_fgsm(input, labels, model, criterion, epsilon, device):
    assert isinstance(model, torch.nn.Module), "Input parameter model is not nn.Module. Check the model"
    assert isinstance(criterion, torch.nn.Module), "Input parameter criterion is no Loss. Check the criterion"
    assert (0 <= epsilon <= 1), "episilon must be 0 <= epsilon <= 1"

    # For calculating gradient
    input_for_gradient = Variable(input, requires_grad=True).to(device)
    out = model(input_for_gradient)
    loss = criterion(out, Variable(labels))

    # Calculate gradient
    loss.backward()

    # Calculate sign of gradient
    signs = torch.sign(input_for_gradient.grad.data)

    # Add
    input_for_gradient.data = input_for_gradient.data + (epsilon * signs)

    return input_for_gradient, signs