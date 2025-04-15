import argparse
import torchvision, torch
from torchvision.transforms import v2
from torcheval import metrics
from itertools import cycle
from time import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
import semisup, models
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('method')
parser.add_argument('--num-labeled', type=int, default=250)
parser.add_argument('--lmbda', type=float, default=1)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--sup-batchsize', type=int, default=16)
parser.add_argument('--unsup-batchsize', type=int, default=112)
parser.add_argument('--frequency_threshold', type=float, default=0.8)
parser.add_argument('--type', type=str, default='cosine')
parser.add_argument('--confidence_threshold', type=float, default=0.95)
parser.add_argument('--mi_threshold', type=float, default=0.20)
parser.add_argument('--mc_dropout_passes', type=int, default=30)

args = parser.parse_args()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data preparation
SEED = 123
transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
])
num_classes = 10
train_dataset = torchvision.datasets.CIFAR10('../data', True, transforms)
test_dataset = torchvision.datasets.CIFAR10('../data', False, transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, 100,
    pin_memory=True, num_workers=4)

# Stratified K-Fold
y = np.array(train_dataset.targets)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def train_on_fold(train_subset, val_subset):
    targets = np.array(train_subset.dataset.targets)[train_subset.indices]
    sup_indices, unsup_indices = train_test_split(
        np.arange(len(train_subset)),
        train_size=args.num_labeled,
        stratify=targets,
        random_state=SEED
    )
    train_sup_dataset = torch.utils.data.Subset(train_subset, sup_indices)
    train_unsup_dataset = torch.utils.data.Subset(train_subset, unsup_indices)
    train_sup_dataloader = torch.utils.data.DataLoader(train_sup_dataset, args.sup_batchsize, shuffle=True, num_workers=4, pin_memory=True)
    train_unsup_dataloader = torch.utils.data.DataLoader(train_unsup_dataset, args.unsup_batchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_subset, 100, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model and method
    model = models.WideResNet()
    model.to(device)
    weak_augment = v2.Compose([v2.RandomCrop(32, 4, padding_mode='reflect'), v2.RandomHorizontalFlip()])
    strong_augment = v2.Compose([v2.RandomCrop(32, 4, padding_mode='reflect'), v2.RandomHorizontalFlip(), v2.RandAugment()])
    method = getattr(semisup, args.method)(model, weak_augment, strong_augment, None, args.frequency_threshold, args.type, args.confidence_threshold, args.mi_threshold, args.mc_dropout_passes, device)

    best_val_acc = 0
    best_model_state = None

    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    # Training loop
    for epoch in range(args.epochs):
        tic = time()
        model.train()
        avg_sup_loss = avg_unsup_loss = 0.0
        semisup_n_iter = max(len(train_sup_dataloader), len(train_unsup_dataloader))
        sup_dataloader_iter = cycle(train_sup_dataloader) if len(train_sup_dataloader) < len(train_unsup_dataloader) else iter(train_sup_dataloader)
        unsup_dataloader_iter = cycle(train_unsup_dataloader) if len(train_unsup_dataloader) < len(train_sup_dataloader) else iter(train_unsup_dataloader)

        for (sup_imgs, sup_labels), (unsup_imgs, _) in zip(sup_dataloader_iter, unsup_dataloader_iter):
            sup_imgs, sup_labels, unsup_imgs = sup_imgs.to(device), sup_labels.to(device), unsup_imgs.to(device)
            sup_loss, unsup_loss = method(epoch, sup_imgs, sup_labels, unsup_imgs)
            total_loss = sup_loss + args.lmbda * unsup_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if ema_model:
                ema_model.update_parameters(model)

            avg_sup_loss += float(sup_loss) / semisup_n_iter
            avg_unsup_loss += float(unsup_loss) / semisup_n_iter

        toc = time()
        print(f'Train - Epoch {epoch+1}/{args.epochs} - {toc-tic:.1f}s - Avg sup loss: {avg_sup_loss} - Avg unsup loss: {avg_unsup_loss}')

        # Validation
        eval_model = model if args.method == 'Supervised' else ema_model
        eval_model.eval()
        val_acc = metrics.MulticlassAccuracy(device=device)
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = eval_model(inputs)
                val_acc.update(outputs, targets)
        val_acc_value = val_acc.compute().item()
        print(f'Fold Validation - Epoch {epoch + 1}/{args.epochs} - Val_Accuracy: {val_acc_value}')

        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            best_model_state = ema_model.module.state_dict() if hasattr(ema_model, 'module') else ema_model.state_dict()

    if args.method == 'Supervised'
        torch.optim.swa_utils.update_bn(train_sup_dataloader, ema_model, device)
    else:
        combined_dataset = torch.utils.data.ConcatDataset([train_sup_dataset, train_unsup_dataset])
        combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=args.sup_batchsize, shuffle=True, num_workers=4, pin_memory=True)
        torch.optim.swa_utils.update_bn(combined_dataloader, ema_model, device)
 
    return best_model_state

# Iterate through folds
best_models = []
for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_dataset)), y)):
    print(f"Fold {fold}:")
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    best_model_state = train_on_fold(train_subset, val_subset)
    best_models.append(best_model_state)

highest_test_acc = 0
best_model_state = None

# Evaluate on test set using the best models from each fold
for fold, model_state in enumerate(best_models):
    model = models.WideResNet()
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    acc = metrics.MulticlassAccuracy(device=device)
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc.update(outputs, targets)
    test_acc_value = acc.compute().item()

    print(f'Test Accuracy for Fold {fold}: {test_acc_value}')

    if test_acc_value > highest_test_acc:
        highest_test_acc = test_acc_value
        best_model_state = model_state

if best_model_state is not None:
    torch.save(best_model_state, args.output)
    print(f'Model with highest test accuracy saved. Accuracy: {highest_test_acc}')
else:
    print('No model was saved.')