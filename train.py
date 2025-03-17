import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('method')
parser.add_argument('--num-labeled', type=int, default=1000)
parser.add_argument('--lmbda', type=float, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--sup-batchsize', type=int, default=16)
parser.add_argument('--unsup-batchsize', type=int, default=112)
args = parser.parse_args()

import torchvision, torch
from torchvision.transforms import v2
from torcheval import metrics
from itertools import cycle
from time import time
import semisup, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################## DATA ##############################################

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
])
num_classes = 10
train_dataset = torchvision.datasets.CIFAR10('/data/toys', True, transforms)
val_dataset = torchvision.datasets.CIFAR10('/data/toys', False, transforms)
val_dataloader = torch.utils.data.DataLoader(val_dataset, args.sup_batchsize+args.unsup_batchsize,
    pin_memory=True, num_workers=4)

# split labeled / unlabeled
SEED = 123
generator = torch.Generator().manual_seed(SEED)
train_sup_dataset, train_unsup_dataset = torch.utils.data.random_split(train_dataset, [args.num_labeled, len(train_dataset) - args.num_labeled], generator)
marginal_distribution = torch.bincount(torch.tensor([y for _, y in train_sup_dataset]), minlength=num_classes).to(device) / len(train_sup_dataset)
train_sup_dataloader = torch.utils.data.DataLoader(train_sup_dataset, args.sup_batchsize, True, num_workers=4, pin_memory=True)
train_unsup_dataloader = torch.utils.data.DataLoader(train_unsup_dataset, args.unsup_batchsize, True, num_workers=4, pin_memory=True)

############################################## MODEL ##############################################

model = models.WideResNet()
model.to(device)

############################################## METHODS ##############################################

weak_augment = v2.Compose([
    v2.RandomCrop(32, 4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),
])
strong_augment = v2.Compose([
    v2.RandomCrop(32, 4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),
    v2.RandAugment(),
])

method = getattr(semisup, args.method)
method = method(model, weak_augment, strong_augment, marginal_distribution)

############################################## TRAIN ##############################################

ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
optimizer = torch.optim.AdamW(model.parameters())

for epoch in range(args.epochs):
    # train
    tic = time()
    model.train()
    avg_sup_loss = avg_unsup_loss = 0
    semisup_n_iter = max(len(train_sup_dataloader), len(train_unsup_dataloader))
    sup_dataloader_iter = cycle(train_sup_dataloader) if len(train_sup_dataloader) < len(train_unsup_dataloader) else iter(train_sup_dataloader)
    unsup_dataloader_iter = cycle(train_unsup_dataloader) if len(train_unsup_dataloader) < len(train_sup_dataloader) else iter(train_unsup_dataloader)

    for (sup_imgs, sup_labels), (unsup_imgs, _) in zip(sup_dataloader_iter, unsup_dataloader_iter):
        sup_imgs, sup_labels = next(sup_dataloader_iter)
        unsup_imgs = next(unsup_dataloader_iter)
        sup_imgs = sup_imgs.to(device)
        sup_labels = sup_labels.to(device)
        unsup_imgs = unsup_imgs[0].to(device)

        sup_loss = torch.nn.functional.cross_entropy(model(weak_augment(sup_imgs)), sup_labels)
        unsup_loss = method(epoch, sup_imgs, sup_labels, unsup_imgs)
        total_loss = sup_loss + args.lmbda*unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if ema_model:
            ema_model.update_parameters(model)

        avg_sup_loss += float(sup_loss) / semisup_n_iter
        avg_unsup_loss += float(unsup_loss) / semisup_n_iter

    toc = time()
    print(f'Train - Epoch {epoch+1}/{args.epochs} - {toc-tic:.1f}s - Avg sup loss: {avg_sup_loss} - Avg unsup loss: {avg_unsup_loss}')

    # evaluate
    eval_model = model if args.method == 'Supervised' else ema_model
    eval_model.eval()
    acc = metrics.MulticlassAccuracy(device=device)
    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        acc.update(outputs, targets)
    print(f'Test  - Epoch {epoch+1}/{args.epochs} - Accuracy: {acc.compute().item()}')

torch.optim.swa_utils.update_bn(train_sup_dataloader, ema_model, device)
model = model if args.method == 'Supervised' else ema_model.module
torch.save(model.cpu(), args.output)
