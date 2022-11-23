import numpy as np
from torchvision.utils import make_grid

from ada_hsi.spixelnet.train_util import init_spixel_grid, update_spixl_map
from ada_hsi.spixelnet.loss import compute_semantic_pos_loss, build_LABXY_feat

from ada_hsi.utils import recons_loss, ideal_spc

import torch
import os
from torch.nn.functional import one_hot


def psnr(y_true, y_est):
    return 10 * torch.log10(1 / torch.mean( (y_true - y_est ) ** 2, dim=(1, 2, 3)))


def get_decimation_matrix(prob, spixelID):
    curr_spixl_map =  update_spixl_map(spixelID, prob)
    curr_spixl_map = curr_spixl_map.long().squeeze(1)
    return curr_spixl_map

def spc_estimation(y_true, prob, spixelID, n_spixels):
    curr_spixl_map = update_spixl_map(spixelID, prob).float()
    curr_spixl_map = curr_spixl_map.squeeze(1)
    curr_spixl_map /= torch.max(curr_spixl_map)
    curr_spixl_map = curr_spixl_map * (n_spixels - 1)
    curr_spixl_map = curr_spixl_map.long()
    decimation_matrix = one_hot(curr_spixl_map, num_classes=n_spixels)
    decimation_matrix = decimation_matrix.permute(0, 3, 1, 2)
    y_est = ideal_spc(y_true, decimation_matrix)
    return y_est

def spixel_psnr(y_true, prob, spixelID, n_spixels):
    y_true = y_true.detach()
    prob = prob.detach()
    spixelID = spixelID.detach()

    y_est = spc_estimation(y_true, prob, spixelID, n_spixels)
    return torch.mean( psnr(y_true, y_est) )

def save_checkpoint(state, is_best, save_path):
    if is_best:
        torch.save(state, save_path)


class TrainingLoop():

    def __init__(self, model, loss_fn, optimizer, train_loader, test_loader, spxiel_args, n_spixels, metrics=None):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.downsize = spxiel_args['downsize']
        self.metrics = metrics
        self.n_spixels = n_spixels

        val_spixel_args = spxiel_args.copy()
        val_spixel_args['batch_size'] = 1

        spixelID, XY_feat_stack = init_spixel_grid(spxiel_args)
        val_spixelID, val_XY_feat_stack = init_spixel_grid(val_spixel_args)

        self.spixelID = spixelID
        self.XY_feat_stack = XY_feat_stack

        self.val_spixelID = val_spixelID
        self.val_XY_feat_stack = val_XY_feat_stack
        self.slic_weight = 1e-8

        
    def train_one_epoch(self, freq=3):

        running_loss = 0.0
        running_psnr = 0.0

        for i, data in enumerate(self.train_loader, 0):

            img_path, sample = data
            gs, ir, D = sample

            LABXY_feat_tensor = build_LABXY_feat(D, self.XY_feat_stack)
            prob, _ = self.model( (ir, gs, D) )

            # Compute SlIC loss
            slic_loss, _, _ = compute_semantic_pos_loss(prob, LABXY_feat_tensor, kernel_size=self.downsize)
            # Compute reconstruction loss
            r_loss = recons_loss(ir, prob, self.downsize, self.loss_fn)
            # Compute total loss
            loss = self.slic_weight*slic_loss + r_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_psnr += spixel_psnr(ir, prob, self.spixelID, self.n_spixels).item()
        
        return running_loss / len(self.train_loader), running_psnr / len(self.train_loader)


    def evaluate(self):
        
        N = len(self.test_loader)
        running_val_loss = 0.0
        running_val_psnr = 0.0

        for i, data in enumerate(self.test_loader, 0):

            img_path, sample = data
            gs, ir, D = sample

            LABXY_feat_tensor = build_LABXY_feat(D, self.val_XY_feat_stack)
            prob, _ = self.model( (ir, gs, D) )

            # Compute SlIC loss
            slic_loss, _, _ = compute_semantic_pos_loss(prob, LABXY_feat_tensor, kernel_size=self.downsize)
            # Compute reconstruction loss
            r_loss = recons_loss(ir, prob, self.downsize, self.loss_fn)
            # Compute total loss
            loss = self.slic_weight*slic_loss + r_loss

            running_val_loss += loss.item()
            running_val_psnr += spixel_psnr(ir, prob, self.val_spixelID, self.n_spixels).item()
            
        return running_val_loss / N, running_val_psnr / N

    
    def fit(self, n_epochs, save_path):

        best_EPE = -1
        val_loss = 0.0 
        val_psnr = 0.0
        for epoch in range(n_epochs):
            train_loss, psnr = self.train_one_epoch()
            val_loss, val_psnr = self.evaluate()
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, PSNR: {psnr:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.4f}')

            if best_EPE < 0:
                best_EPE = val_psnr
            
            is_best = val_psnr > best_EPE
            best_EPE = max(val_psnr, best_EPE)

            rec_dict = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_EPE': best_EPE,
                'optimizer': self.optimizer.state_dict(),
             }

            save_checkpoint(rec_dict, is_best, save_path)
    
    def predict(self, sample):
        gs, ir, D = sample
        prob, _ = self.model( (ir, gs, D) )
        decimation_matrix = get_decimation_matrix(prob, self.val_spixelID)
        return spc_estimation(ir, prob, self.val_spixelID, self.n_spixels), decimation_matrix

           













            
