# encoding = utf-8

import torch
import torch.nn as nn
from torch.autograd import Function


class CrossEntropyFunc(Function):

    @staticmethod
    def forward(ctx, feature, one_hot_label):
        batch_size = feature.size()[0]
        feature_max, _ = torch.max(feature, dim=1, keepdim=True)
        print(feature_max)
        feature_exp = torch.exp(feature - feature_max)
        feature_exp_sum = torch.sum(feature_exp, dim=1, keepdim=True)
        # softmax = exp(i) / sum(exp(j)) j=0,1,2...
        feature_softmax = torch.div(feature_exp, feature_exp_sum)
        ctx.save_for_backward(feature_softmax, one_hot_label)

        # compute loss
        feature_log_softmax = -torch.log(feature_softmax)
        probability = feature_log_softmax * one_hot_label
        loss = torch.sum(probability) / batch_size
        return loss

    @staticmethod
    def backward(ctx, grad_outputs):
        feature_softmax, one_hot_label = ctx.saved_tensors
        batch_size = feature_softmax.size()[0]
        # compute feature grad: softmax - one-hot-label
        feature_grad = (feature_softmax - one_hot_label) / batch_size
        print('feature grad: ', feature_grad, batch_size)
        return feature_grad, None


def cross_entropy(feature, one_hot_label):
    return CrossEntropyFunc.apply(feature, one_hot_label)


if __name__ == "__main__":
    batch_sizes = 2
    class_num = 3
    x = torch.tensor([[1.2, 2.3, 3.2], [1.25, 2.34, 4.5]], requires_grad=True)
    label = torch.tensor([[2], [1]])
    label1 = torch.tensor([2, 1])
    y = torch.zeros(batch_sizes, class_num).scatter_(1, label, 1)
    print(x, y)

    loss = cross_entropy(x, y)
    loss.backward(retain_graph=True)
    print('loss: ', loss, 'x grad :', x.grad)

    x.grad.data.zero_()
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(x, label1)
    loss.backward()
    print('loss: ', loss, 'x grad :', x.grad)
