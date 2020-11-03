import math
import numpy as np
import re
import torch
import wandb
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm.auto import tqdm, trange


class Trainer:
    def __init__(self, project_name, model, device, dataloader, classes, criterion,
                 optimizer=None, optimizer_params={"lr": 1e-3}, optimizer_name="sgd", scheduler=None,
                 lr_find=True, lr_finder_params={"start_lr": 1e-8, "end_lr": 1, "num_iter": 100, "step_mode": "exp"},
                 max_epoch=5, max_step=None, gradient_accumulation=1,
                 valloader=None, logging_steps=100, validation_steps=500):
        wandb.init(project=project_name)
        config = wandb.config

        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.valloader = valloader
        self.classes = classes
        self.criterion = criterion
        config.logging_steps = self.logging_steps = logging_steps
        config.validation_steps = self.validation_steps = validation_steps
        config.gradient_accumulation = self.gradient_accumulation = gradient_accumulation
        config.max_epoch = self.max_epoch = max_epoch if max_step is None else int(
            math.ceil(max_step * gradient_accumulation / len(dataloader)))
        config.max_step = self.max_step = int(max_epoch * len(dataloader) /
                                              gradient_accumulation) if max_step is None else max_step

        self.optimizer = self.__class__.get_optimizer(
            self.model, optimizer_name, optimizer_params) if optimizer is None else optimizer

        suggested_lr = None

        if lr_find:
            from torch_lr_finder import LRFinder
            config.lr_finder_params = lr_finder_params
            lr_finder = LRFinder(self.model, self.optimizer,
                                 self.criterion, device=self.device)
            lr_finder.range_test(self.dataloader, val_loader=self.valloader, **lr_finder_params,
                                 accumulation_steps=gradient_accumulation)
            _, suggested_lr = lr_finder.plot()

        print(suggested_lr)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=optimizer_params.get(
            "lr", 0.01) if suggested_lr is None else suggested_lr, total_steps=self.max_step) if scheduler is None else scheduler

        # wandb logging
        config.batch_size = dataloader.batch_size
        config.optimizer = re.sub(r"\s+", " ", str(self.optimizer))
        config.scheduler = re.sub(
            r"\s+", " ", str(self.scheduler.state_dict()))

    @staticmethod
    def get_optimizer(model, optimizer_name, params_dict):
        optims_dict = {
            "adam": lambda: optim.Adam(model.parameters(), **params_dict),
            "sgd": lambda: optim.SGD(model.parameters(), **params_dict),
        }
        return optims_dict.get(optimizer_name, optims_dict["sgd"])()

    def train(self):
        global_steps = 0
        actual_steps = 0
        self.model.zero_grad()
        running_loss = 0
        for epoch in range(self.max_epoch):  # loop over the dataset multiple times
            for i, (inputs, labels) in enumerate(tqdm(self.dataloader, desc="Step")):
                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                loss.backward()
                running_loss += loss.item()
                actual_steps += 1

                if actual_steps % self.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    global_steps += 1

                    if global_steps % self.logging_steps == 0:
                        current_loss = running_loss / self.logging_steps / self.gradient_accumulation
                        wandb.log({
                            'training_loss': current_loss,
                            'learning_rate': self.scheduler.get_last_lr()[0],
                        }, step=global_steps)
                        # print('[%d, %5d] loss: %.3f' % (epoch + 1, global_steps, current_loss))
                        running_loss = 0

                    if self.validation_steps > 0 and global_steps % self.validation_steps == 0:
                        if self.valloader is not None:
                            y_val_pred, y_val_actual, y_val_loss = self.evaluation(
                                self.valloader)
                            wandb.log({
                                'val_loss': y_val_loss,
                                'val_accuracy': accuracy_score(y_val_actual, y_val_pred),
                                'roc': wandb.plots.ROC(y_val_actual, self.to_binary(y_val_pred), self.classes),
                                'pr': wandb.plots.precision_recall(y_val_actual, self.to_binary(y_val_pred), self.classes),
                            }, step=global_steps)

                    if global_steps == self.max_step:
                        break

        print("Training completed")

    def evaluation(self, dataloader):
        self.model.eval()
        y_pred = []
        y_actual = []
        eval_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                calculated_loss = self.criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            eval_loss += calculated_loss.item()
            y_pred.append(predicted)
            y_actual.append(labels)

        return torch.cat(y_pred).cpu().numpy(), torch.cat(y_actual).cpu().numpy(), eval_loss / len(dataloader)

    def to_binary(self, labels):
        return np.asarray([[1 if class_index == label else 0 for class_index in range(len(self.classes))]
                           for label in labels])
