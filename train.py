import torch
from torch.utils.data import DataLoader, Dataset
from imagenet_dataloader import imagenet_get_dataset
import mlflow
from torchmetrics import Accuracy
import torchvision

class TrainPipeline:
    def __init__(
        self,
        run_name: str,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        val_dataset: Dataset,
        model, device,
        n_classes: int,
        batch_size_labeled: int,
        batch_size_unlabeled: int,
        n_epochs: int,
        tau: float,
        lambda_u: float
    ):
        self._run_name = run_name
        self._labeled_dataset = labeled_dataset
        self._unlabeled_dataset = unlabeled_dataset
        self._val_dataset = val_dataset
        self._model = model
        self._device = device
        self._n_classes = n_classes
        self._batch_size_labeled = batch_size_labeled
        self._batch_size_unlabeled = batch_size_unlabeled
        self._n_epochs = n_epochs
        self._tau = tau
        self._lambda_u = lambda_u
        self._labeled_dataloader = DataLoader(self._labeled_dataset, batch_size=self._batch_size_labeled, shuffle=True)
        self._unlabeled_dataloader = DataLoader(self._unlabeled_dataset, batch_size=self._batch_size_unlabeled, shuffle=True)
        self._val_dataloader = DataLoader(self._val_dataset, batch_size=100, shuffle=False)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        self._losses = {
            'train_loss': 0,
            'val_loss': 0
        }
        self._metrics = {'val_accuracy': Accuracy(task="multiclass", num_classes=self._n_classes).to(self._device)}



    def _compute_unlabeled_loss(self, weak_preds, strong_preds):
        """
        Computes the unlabeled loss
        """
        pseudo_labels = weak_preds.argmax(dim=1)
        confidence = weak_preds.max(dim=1)[0]

        mask = confidence >= self._tau
        num_selected = mask.sum().item()

        if num_selected > 0:
            strong_preds = strong_preds[mask]
            pseudo_labels = pseudo_labels[mask]

            loss_unlabeled = torch.nn.CrossEntropyLoss()(strong_preds, pseudo_labels)
        else:
            loss_unlabeled = torch.tensor(0.0, device=self._device)

        return loss_unlabeled


    def run(self):
        scaler = torch.amp.GradScaler(device='cuda')
        with mlflow.start_run(run_name=self._run_name):
            mlflow.pytorch.autolog()
            mlflow.log_param('n_epochs', self._n_epochs)
            mlflow.log_param('tau', self._tau)
            mlflow.log_param('lambda_u', self._lambda_u)
            best_accuracy = 0.0
            for epoch in range(self._n_epochs):
                self._model.train()
                num_batches = 0
                for (labeled_data, labeled_target), (weak_unlabeled_data, strong_unlabeled_data) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
                    labeled_data, labeled_target, weak_unlabeled_data, strong_unlabeled_data = labeled_data.to(self._device), labeled_target.to(self._device), weak_unlabeled_data.to(self._device), strong_unlabeled_data.to(self._device)
                    with torch.amp.autocast(device_type='cuda'):
                        labeled_pred = self._model(labeled_data)
                        loss_labeled = torch.nn.CrossEntropyLoss()(labeled_pred, labeled_target)

                        weak_pred = torch.softmax(self._model(weak_unlabeled_data), dim=1)
                        strong_pred = self._model(strong_unlabeled_data)
                        loss_unlabeled = self._compute_unlabeled_loss(weak_pred, strong_pred)

                        loss = loss_labeled + self._lambda_u * loss_unlabeled                    
                    
                    self._model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self._optimizer)
                    scaler.update()

                    num_batches += 1
                    self._losses['train_loss'] += loss.item()

                self._model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        val_num_batches = 0
                        for (images, annotations) in self._val_dataloader:
                            val_num_batches += 1
                            images, annotations = images.to(self._device), annotations.to(self._device)
                            
                            pred = self._model(images)
                            val_loss = torch.nn.CrossEntropyLoss()(pred, annotations)
                            self._losses['val_loss'] += val_loss.item()
                            self._metrics['val_accuracy'].update(pred, annotations)


                val_accuracy = self._metrics['val_accuracy'].compute().item()

                if val_accuracy > best_accuracy:
                    torch.save(self._model.state_dict(), f"{self._run_name}.pth")
                    print(f"Model saved with val Accuracy: {val_accuracy:.4f}; Epoch: {epoch+1}")
                    best_accuracy = val_accuracy

                for loss_name, value in self._losses.items():
                    _n_batches = val_num_batches if loss_name.startswith('val_') else num_batches
                    mlflow.log_metric(loss_name, value / _n_batches, step=epoch+1)
                    print("Loss: ", loss_name, value / _n_batches)
                    self._losses[loss_name] = 0

                for metric_name, metric in self._metrics.items():
                    mlflow.log_metric(metric_name, metric.compute().item(), step=epoch+1)
                    metric.reset()


if __name__ == '__main__':

    # model.load_state_dict(torch.load("supervised.pth", map_location=device))

    labeled_training_dataset, unlabeled_training_dataset, val_dataset = imagenet_get_dataset(n_labeled=13000)
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    TrainPipeline(
        run_name="fixmatch",
        labeled_dataset=labeled_training_dataset,
        unlabeled_dataset=unlabeled_training_dataset,
        val_dataset=val_dataset,
        model=model,
        device=device,
        n_classes=1000,
        batch_size_labeled=16,
        batch_size_unlabeled=80,
        n_epochs=300,
        tau=0.95,
        lambda_u=1.0
    ).run()