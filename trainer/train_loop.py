import gc
import wandb, optuna
import torch
import numpy as np
from tqdm.auto import tqdm

from torch.optim.swa_utils import update_bn
from configuration import CFG
from trainer import *
from trainer.trainer_utils import get_name
from utils.helper import class2dict

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: any) -> None:
    """ Base Trainer Loop Function """
    fold_list = [i for i in range(cfg.n_folds)]
    for fold in tqdm(fold_list[2:]):
        print(f'============== {fold}th Fold Train & Validation ==============')
        wandb.init(
            project=cfg.name,
            name=f'[{cfg.model_arch}]' + f'fold{fold}/' + cfg.model,
            config=class2dict(cfg),
            group=f'max_length_{cfg.max_len}/{cfg.model}',
            job_type='train',
            entity="qcqced"
        )
        early_stopping = EarlyStopping(mode=cfg.stop_mode)
        early_stopping.detecting_anomaly()

        val_score_max, fold_swa_loss = -np.inf, []
        train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
        loader_train, loader_valid, train, valid_labels = train_input.make_batch(fold)
        model, swa_model, criterion, val_criterion, val_metrics, optimizer,\
            lr_scheduler, swa_scheduler, awp = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss, grad_norm, lr = train_input.train_fn(
                loader_train, model, criterion, optimizer, lr_scheduler,
                epoch, awp, swa_model, cfg.swa_start, swa_scheduler
            )
            valid_loss, epoch_score = train_input.valid_fn(
                loader_valid, model, val_criterion, val_metrics, valid_labels
            )
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Valid Loss': valid_loss,
                '<epoch> Pearson Score': epoch_score,
                '<epoch> Gradient Norm': grad_norm,
                '<epoch> lr': lr
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(valid_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Pearson Score: {np.round(epoch_score, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Gradient Norm: {np.round(grad_norm, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] lr: {lr}')
            if val_score_max <= epoch_score:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {epoch_score:.4f}) Save Parameter')
                print(f'Best Score: {epoch_score}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}fold{fold}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                val_score_max = epoch_score
            # Check if Trainer need to Early Stop
            early_stopping(epoch_score)
            if early_stopping.early_stop:
                break
            del train_loss, valid_loss, epoch_score, grad_norm, lr
            gc.collect(), torch.cuda.empty_cache()

        if not early_stopping.early_stop:
            update_bn(loader_train, swa_model)
            swa_loss, swa_valid_score = train_input.swa_fn(
                loader_valid, swa_model, val_criterion, val_metrics, valid_labels
            )
            print(f'Fold[{fold}/{fold_list[-1]}] SWA Loss: {np.round(swa_loss, 4)}')
            print(f'Fold[{fold}/{fold_list[-1]}] SWA Pearson Score: {np.round(swa_valid_score, 4)}')

            if val_score_max <= swa_valid_score:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {swa_valid_score:.4f}) Save Parameter')
                print(f'Best Score: {swa_valid_score}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}SWA_fold{fold}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                wandb.log({'<epoch> Valid Loss': swa_loss})

        wandb.finish()


def mpl_loop(cfg: any) -> None:
    """ MPL Trainer Loop Function """
    wandb.init(project=cfg.name,
               name=f'[{cfg.model_arch}]' + '/Meta Pseudo Label/' + cfg.model,
               config=class2dict(cfg),
               group=cfg.model,
               job_type='train',
               entity="qcqced")
    val_score_max = 0.4536
    train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
    s_loader_train, s_train, p_loader_train, p_loader_valid, p_train = train_input.make_batch()
    t_model, s_model, criterion, t_optimizer, \
        s_optimizer, t_scheduler, s_scheduler, save_parameter = train_input.model_setting(
            len(s_train), len(p_train)
        )
    s_valid_loss = torch.Tensor([0.4536]).to(cfg.device)
    for epoch in range(cfg.epochs):
        print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
        t_train_loss, s_train_loss, t_lr, s_lr = train_input.train_fn(
            t_model, s_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler,
            s_loader_train, p_loader_train, s_valid_loss
        )
        s_valid_loss, s_valid_losses = train_input.valid_fn(
            p_loader_valid, s_model, criterion
        )
        wandb.log({
            '<epoch> Teacher Train Loss': t_train_loss,
            '<epoch> Student Train Loss': s_train_loss,
            '<epoch> Student Validation Loss': s_valid_losses,
            '<epoch> Teacher lr': t_lr,
            '<epoch> Student lr': s_lr,
        })
        print(f'[{epoch + 1}/{cfg.epochs}] Teacher Train Loss: {np.round(t_train_loss, 4)}')
        print(f'[{epoch + 1}/{cfg.epochs}] Student Train Loss: {np.round(s_train_loss, 4)}')
        print(f'[{epoch + 1}/{cfg.epochs}] Student Validation Loss: {np.round(s_valid_losses, 4)}')

        if val_score_max >= s_valid_losses:
            print(f'[Update] Valid Score : ({val_score_max:.4f} => {s_valid_losses:.4f}) Save Parameter')
            print(f'Best Score: {s_valid_losses}')
            torch.save(t_model.state_dict(),
                       f'{cfg.checkpoint_dir}{cfg.state_dict}MPL_Teacher_{get_name(cfg)}_state_dict.pth')
            torch.save(s_model.state_dict(),
                       f'{cfg.checkpoint_dir}{cfg.state_dict}MPL_Student_{get_name(cfg)}_state_dict.pth')
            val_score_max = s_valid_losses

        del t_train_loss, s_train_loss, s_valid_losses, t_lr, s_lr
        gc.collect(), torch.cuda.empty_cache()

    wandb.finish()


def hyper_params_tuning(cfg: any, trial: any) -> None:
    """ Optuna Tuning Hyper-Params Loop Function """
    wandb.init(project=cfg.name,
               name=f'[{cfg.model_arch}]' + '/Meta Pseudo Label/' + cfg.model,
               config=class2dict(cfg),
               group=cfg.model,
               job_type='train',
               entity="qcqced")

    wandb.finish()
