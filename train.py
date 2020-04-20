import torch
import torch.nn as nn
from dataset import make_dataset
from sklearn.metrics import f1_score
import pickle
from resnext import resnet101
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

big_model = resnet101(sample_size=112, sample_duration=32, num_classes=27, shortcut_type='A')
checkpoint = torch.load('./pretrain/models/jester_resnext_101_RGB_32.pth')
weights = OrderedDict()
for w_name in checkpoint['state_dict']:
    _w_name = '.'.join(w_name.split('.')[1:])
    weights[_w_name] = checkpoint['state_dict'][w_name]
big_model.load_state_dict(weights)
big_model.cuda()

model = 
try:
    checkpoint = torch.load('./best_classifier_checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
except:
    pass
model.cuda()

train_ds, test_ds = make_dataset()
loss_fn = nn.CrossEntropyLoss()
def SoftmaxWithTemperature(temp=1, dim=1):
    def fn(x):
        exp = torch.exp(x/temp)
        denom = torch.sum(exp, dim)
        return exp/denom
    return fn
act_fn = SoftmaxWithTemperature(temp=10)
learning_rate = 0.001
clip_value = 0.05
epochs = 100
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
hard_test_losses = []
soft_test_losses = []
hard_test_f1s = []
soft_test_f1s = []
hard_test_accuracy = []
soft_test_accuracy = []
with torch.no_grad():
    avg_train_loss = 0
    for batch, _, target in tqdm(train_ds):
        batch = batch.cuda()
        target = target.cuda()
        pred = big_model(batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, target)
        avg_train_loss += loss.item()
    avg_train_loss /= len(train_ds)
    avg_test_loss = 0
    avg_test_f1 = 0
    acg_test_accuracy = 0
    for batch, _, target in tqdm(test_ds):
        batch = batch.cuda()
        target = target.cuda()
        pred = big_model(batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, target)
        avg_test_loss += loss.item()
        np_pred = pred.cpu().detach().numpy()
        pred_idx = np.argmax(np_pred, axis=1).reshape(-1)
        np_target = target.cpu().detach().numpy().reshape(-1)
        avg_test_f1 += f1_score(pred_idx, np_target)
        avg_test_accuracy += np.sum(pred_idx == np_target)/len(np_target)
    avg_test_f1 /= len(test_ds)
    avg_test_accuracy /= len(test_ds)
    print('EVAL BIG MODEL RESULTS:\nTRAIN LOSS: %f\tTEST LOSS: %f\tTEST F1: %f\tTEST ACC: %f\n%s' % (avg_train_loss, avg_test_loss, avg_test_f1, avg_test_accuracy, '-'*80))
for e in range(epochs):
    train_ds.shuffle()
    avg_train_loss = 0
    for batch, big_batch, hard_target in tqdm(train_ds):
        optim.zero_grad()
        batch = batch.cuda()
        big_batch = big_batch.cuda()
        hard_target = hard_target.cuda()
        pred = big_model(batch)
        soft_target = act_fn(pred)
        pred = model(big_batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, soft_target)
        avg_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optim.step()
    avg_train_loss /= len(train_ds)
    train_losses.append(avg_train_loss)
    avg_hard_test_loss = 0
    avg_soft_test_loss = 0
    avg_hard_test_f1 = 0
    avg_soft_test_f1 = 0
    avg_hard_test_accuracy = 0
    avg_soft_test_accuracy = 0
    with torch.no_grad():
        for batch, big_batch, hard_target in tqdm(test_ds):
            batch = batch.cuda()
            big_batch = big_batch.cuda()
            hard_target = hard_target.cuda()
            pred = big_model(batch)
            soft_target = act_fn(pred)
            pred = model(big_batch)
            pred = act_fn(pred)
            avg_hard_test_loss = loss_fn(pred, hard_target).item()
            avg_soft_test_loss = loss_fn(pred, soft_target).item()
            np_pred = pred.cpu().detach().numpy()
            pred_idx = np.argmax(np_pred, axis=1).reshape(-1)
            np_hard_target = hard_target.cpu().detach().numpy().reshape(-1)
            np_soft_target = np.argmax(soft_target.cpu().detach().numpy(), axis=1).reshape(-1)
            avg_hard_test_f1 += f1_score(pred_idx, np_hard_target)
            avg_soft_test_f1 += f1_score(pred_idx, np_soft_target)
            avg_hard_test_accuracy += np.sum(pred_idx == np_hard_target)/len(np_hard_target)
            avg_soft_test_accuracy += np.sum(pred_idx == np_soft_target)/len(np_soft_target)
        avg_hard_test_loss /= len(test_ds)
        avg_soft_test_loss /= len(test_ds)
        avg_hard_test_f1 /= len(test_ds)
        avg_soft_test_f1 /= len(test_ds)
        avg_hard_test_accuracy /= len(test_ds)
        avg_soft_test_accuracy /= len(test_ds)
        hard_test_losses.append(avg_hard_test_loss)
        soft_test_losses.append(avg_soft_test_loss)
        hard_test_f1s.append(avg_hard_test_f1)
        soft_test_f1s.append(avg_soft_test_f1)
        hard_test_accuracy.append(avg_hard_test_accuracy)
        soft_test_accuracy.append(avg_soft_test_accuracy)
        print('EPOCH: %d\tTRAIN LOSS: %f\tTEST LOSS: (%f|%f)\tTEST F1: (%f|%f)\tTEST ACC: (%f|%f)\n' % (e + 1, train_losses[-1], soft_test_losses[-1], hard_test_losses[-1], soft_test_f1s[-1], hard_test_f1s[-1], soft_test_accuracy[-1], hard_test_accuracy[-1]))
        state_dict = model.state_dict()
        torch.save(state_dict, 'classifier_checkpoint.pth.tar')
        if avg_hard_test_f1 == max(hard_test_f1s):
            torch.save(state_dict, 'best_classifier_checkpoint.pth.tar')
with open('./train_losses.dat', 'wb') as fw:
    pickle.dump(train_losses, fw)
with open('./hard_test_losses.dat', 'wb') as fw:
    pickle.dump(hard_test_losses, fw)
with open('./soft_test_losses.dat', 'wb') as fw:
    pickle.dump(soft_test_losses, fw)
with open('./hard_test_f1s.dat', 'wb') as fw:
    pickle.dump(hard_test_f1s, fw)
with open('./soft_test_f1s.dat', 'wb') as fw:
    pickle.dump(soft_test_f1s, fw)
with open('./hard_test_accuracy.dat', 'wb') as fw:
    pickle.dump(hard_test_accuracy, fw)
with open('./soft_test_accuracy.dat', 'wb') as fw:
    pickle.dump(soft_test_accuracy, fw)
