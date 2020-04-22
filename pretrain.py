import torch
import torch.nn as nn
from mobilenetv2 import get_model, filter_state_dict_by_layer
import imgaug.augmenters as iaa
from dataset import make_pretrain_datasets
import shutil

class args:
    lr = 0.01
    epochs = 3
    eval_freq = 13900
    print_freq = 7000
    lmbd = 1.0
    ed_lmbd = 1.0
    batch_size = 8
    noise_magnitude = 0.1
    pretrain_depth = 1

def _lr_step(epoch, base_lr):
    return base_lr * (0.1 ** (epoch // 30))

def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = _lr_step(epoch, base_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, acc, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_%s' % filename)

def get_ed_reg_loss(model):
    e_norm = torch.tensor(0.0).cuda()
    d_norm = torch.tensor(0.0).cuda()
    features_state = filter_state_dict_by_layer(model.state_dict(), 'features')
    for p_name in features_state:
        if features_state[p_name].requires_grad:
            e_norm += (features_state[p_name].norm())**2
    inv_features_state = filter_state_dict_by_layer(model.state_dict(), 'inv_features')
    for p_name in inv_features_state:
        if inv_features_state[p_name].requires_grad:
            d_norm += (inv_features_state[p_name].norm())**2
    ed_reg_loss = (e_norm**(0.5) - d_norm**(0.5))**2
    return ed_reg_loss

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

model = get_model(num_classes=1, sample_size=224, width_mult=1.0, pretrain_depth=args.pretrain_depth)
if args.pretrain_depth > 1:
    try:
        checkpoint = torch.load('./best_mobilenetv2_%d_checkpoint.pth.tar', map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(checkpoint)
        print('Loaded previous pretrain step weights')
    except:
        print('Training from scratch')
model.classifier.eval()
enc_parameters = 0
dec_parameters = 0
for p in model.features.parameters():
    enc_parameters += p.numel()
for p in model.inv_features.parameters():
    dec_parameters += p.numel()
number_of_parameters = enc_parameters + dec_parameters
print('Encoder: %d, decoder: %d, total: %d' % (enc_parameters, dec_parameters, enc_parameters + dec_parameters))
sometimes = lambda aug: iaa.Sometimes(0.8, aug)
aug = iaa.Sequential([sometimes(iaa.AdditiveGaussianNoise(scale=args.noise_magnitude*255))])
train_ds, test_ds = make_pretrain_datasets(aug=aug, batch_size=args.batch_size)

while model.pretrain_depth < len(model.features):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    best_acc = float('inf')
    for epoch in range(args.epochs):
        losses = AverageMeter()
        precs = AverageMeter()
        ed_losses = AverageMeter()
        grad_norms = AverageMeter()
        model.cuda()
        model.train()
        train_ds.shuffle()
        for i, (input, correct) in enumerate(train_ds):
            model.train()
            input, correct = input.cuda(), correct.cuda()
            optimizer.zero_grad()
            output = model.autoencode(input)
            loss = loss_fn(output, correct)
            loss_reg = torch.tensor(0.0).cuda()
            for param in model.parameters():
                if param.requires_grad:
                    loss_reg += (torch.norm(param))**2
            loss_reg *= args.lmbd/(2*float(number_of_parameters))
            loss += loss_reg
            ed_loss = get_ed_reg_loss(model)*args.ed_lmbd
            loss += ed_loss
            ed_losses.update(ed_loss.item())
            loss.backward()
            with torch.no_grad():
                grad_norm = 0
                for param in model.parameters():
                    if param.requires_grad:
                        grad_norm += (torch.norm(param.grad)**2).item()
            grad_norms.update(grad_norm)
            losses.update(loss.item())
            optimizer.step()
            if i % args.eval_freq == 0 and i > 0:
                with torch.no_grad():
                    model.eval()
                    for j, (einput, ecorrect) in enumerate(test_ds):
                        einput, ecorrect = einput.cuda(), ecorrect.cuda()
                        eoutput = model.autoencode(einput)
                        prec = loss_fn(eoutput, ecorrect)
                        precs.update(prec)
            if i % args.print_freq == 0 and i > 0:
                print('Depth: %d\tEpoch: %d(%d/%d)\tLoss: %f\tED_loss: %f\tPrec: %f\tGrad: %f\n%s' % (args.pretrtain_depth, epoch, i, len(train_ds), losses.avg, ed_losses.avg, precs.avg, grad_norms.avg, '-'*80))
        save_checkpoint({
                'epoch' : epoch + 1,
                'state_dict' : model.state_dict(),
                'prec' : precs.avg,
            }, precs.avg < best_acc, precs.avg, filename='mobilenetv2_%d_checkpoint.pth' % args.pretrain_depth)

        best_acc = min(precs.avg, best_acc)
        adjust_learning_rate(optimizer, epoch, args.lr)
    os.system('rm mobilenetv2_%s_checkpoint.pth.tar' % args.pretrain_depth)
    args.pretrain_depth += 1
    model.pretrain_depth = args.pretrain_depth
