import numpy as np
import torch
from base import BaseTrainer
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, generator, discriminator, criterion, l1_criterion, l1_coef, generator_optimizer, generator_lr_scheduler, discriminator_optimizer, discriminator_lr_scheduler,
                 config, device,
                 data_loader, valid_data_loader=None):
        super().__init__(generator, discriminator, criterion, l1_criterion, l1_coef, [], generator_optimizer, discriminator_optimizer, config)

        self.config = config
        self.device = device
        self.data_loader = data_loader
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False).to(device)

        self.train_metrics = MetricTracker('discriminator_loss', 'generator_loss', 'discriminator_grad_norm', 'generator_grad_norm')
        self.valid_metrics = MetricTracker('discriminator_loss', 'generator_loss')


    def _loss_discriminator(self, A, B, fakeB):
        fake_AB = torch.cat((A, fakeB), dim=1)
        pred_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = self.criterion(pred_fake, torch.tensor(0.0).expand_as(pred_fake))
        real_AB = torch.cat((A, B), dim=1)
        pred_real = self.discriminator(real_AB)
        loss_D_real = self.criterion(pred_real, torch.tensor(1.0).expand_as(pred_real))
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def _loss_generator(self, A, B, fakeB):
        fake_AB = torch.cat((A, fakeB), dim=1)
        pred_fake = self.discriminator(fake_AB)
        loss_G = self.criterion(pred_fake, torch.tensor(1.0).expand_as(pred_fake))
        if self.l1_criterion:
            loss_l1 = self.l1_criterion(fakeB, B) * self.l1_coef
            loss_G += loss_l1
        return loss_G

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.discriminator.train()
        self.train_metrics.reset()
        for batch_idx, (A, B) in enumerate(self.data_loader):
            A, B = A.to(self.device), B.to(self.device)
            fakeB = self.generator(A)
        
            self.set_requires_grad(self.discriminator, True)
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self._loss_discriminator(A, B, fakeB)
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.set_requires_grad(self.discriminator, False)
            self.generator_optimizer.zero_grad()
            generator_loss = self._loss_generator(A, B, fakeB)
            generator_loss.backward()
            self.generator_optimizer.step()

            self.train_metrics.update('discriminator_loss', discriminator_loss.item())
            self.train_metrics.update('generator_loss', generator_loss.item())
            self.train_metrics.update("discriminator_grad_norm", self.get_grad_norm(self.discriminator))
            self.train_metrics.update("generator_grad_norm", self.get_grad_norm(self.generator))

            if batch_idx % self.log_step == 0:
                if self.writer is not None:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                    if self.generator_lr_scheduler is not None:
                        self.writer.add_scalar(
                            "generator_learning_rate", self.generator_lr_scheduler.get_last_lr()[0]
                        )
                    if self.discriminator_lr_scheduler is not None:
                        self.writer.add_scalar(
                            "discriminator_learning_rate", self.discriminator_lr_scheduler.get_last_lr()[0]
                        )
                    for metric_name in self.train_metrics.keys():
                        self.writer.add_scalar(f"{metric_name}", self.train_metrics.avg(metric_name))
                logger_message = "Train Epoch: {} {}".format(epoch, self._progress(batch_idx))
                for metric_name in self.train_metrics.keys():
                    metric_res = self.train_metrics.avg(metric_name)
                    if self.writer is not None:
                        self.writer.add_scalar(f"{metric_name}", metric_res)
                    logger_message += f" {metric_name}: {metric_res:.2f}"
                self.logger.debug(logger_message)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()
        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()
        return log

    @torch.no_grad()
    def _log_image_examples(self, sample_size=5):
        A, B = next(iter(self.valid_data_loader))
        A, B = A[:sample_size].to(self.device), B[:sample_size].to(self.device)
        genB = self.generator(A)
        images = torch.cat([B, genB], dim=-1).cpu().detach().numpy()
        images = np.transpose(images, axes=[0, 2, 3, 1])
        self.writer.add_images("image examples", images)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.discriminator.eval()
        self.valid_metrics.reset()
        self._log_image_examples()
        self.fid.reset()
        with torch.no_grad():
            for batch_idx, (A, B) in enumerate(self.valid_data_loader):
                A, B = A.to(self.device), B.to(self.device)
                fakeB = self.generator(A)
                self.fid.update((B * 255).byte(), real=True)
                self.fid.update((fakeB * 255).byte(), real=False)

                discriminator_loss = self._loss_discriminator(A, B, fakeB)
                generator_loss = self._loss_generator(A, B, fakeB)

                self.valid_metrics.update('discriminator_loss', discriminator_loss.item())
                self.valid_metrics.update('generator_loss', generator_loss.item())
    
        if self.writer is not None:
            self.writer.set_step(epoch * self.len_epoch, mode="val")
            self.writer.add_scalar("fid", self.fid.compute().item())
            for metric_name in self.valid_metrics.keys():
                self.writer.add_scalar(f"{metric_name}", self.valid_metrics.avg(metric_name))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} steps ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
