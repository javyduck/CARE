import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch_geometric.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.optim.lr_scheduler import StepLR
from datasets import get_gcn_dataset  # Assume you have this in your project
from architectures import GCN  # Assume you have this in your project

PRECISION = 1e-6

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Reasoning Training')
    parser.add_argument('--noise_sd', default=0.50, type=float, help="Standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--w', default=0.5, type=float, help="Weight for the classification loss")
    parser.add_argument('--dataset', default='AWA', type=str, help="Dataset for training GCN")
    parser.add_argument('--sample_num', default=10, type=int, help="Number of samples")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size for DataLoader")
    parser.add_argument('--lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay for optimizer")
    parser.add_argument('--step_size', default=40, type=int, help="Step size for scheduler")
    parser.add_argument('--gamma', default=0.1, type=float, help="Gamma for scheduler")
    parser.add_argument('--training_epoch', default=60, type=int, help="Number of training epochs")
    return parser.parse_args()


def E_step(model, batch, sample_num = 10):
    loss = 0
    logits = model(batch)
    x = batch.x.reshape(-1,total_num).clip(min=precision,max=1-precision)
    y = batch.y.reshape(-1,total_num)
    phi = F.softmax(logits[:,:main_num], dim=1)
    phi = torch.cat((phi, F.sigmoid(logits[:,main_num:])),dim=1)
    phi = phi.clip(min=precision,max=1-precision)
    coef = []
    cat_index = []
    for i in range(sample_num):
        with torch.no_grad():
            ### sample from Q ###
            index = Categorical(probs = phi[:,:main_num]).sample().reshape(-1,1)
            sample = torch.zeros_like(phi[:,:main_num])
            sample.scatter_(index=index, dim=1, value=1)
            sample = torch.cat((sample, Bernoulli(probs = phi[:, main_num:]).sample()),dim=1)
            aux_score = neg_indicator(sample @ formula + bias)
            phi_log = (phi.log() * sample).sum(-1) + ((1-phi[:,main_num:]).log() * (1-sample[:,main_num:])).sum(-1)
            ### main score + knowledge score ###
            tmp = phi_log - ((x/(1-x)).log() * sample).sum(-1) - (aux_score * model.w.exp()).sum(-1)
            coef.append(tmp.unsqueeze(0))
        cat_index.append(((phi.log() * sample).sum(-1) + ((1-phi[:,main_num:]).log() * (1-sample[:,main_num:])).sum(-1)).unsqueeze(0))
    cat_index = torch.cat(cat_index)
    coef = torch.cat(coef)
    coef -= coef.mean(0)
    loss += (coef * cat_index).mean() - args.w * ((phi.log() * y).sum(-1)+((1-phi[:,main_num:]).log() * (1-y[:,main_num:])).sum(-1)).mean()
    return loss

def M_step(model, batch, lr, sample_num = 10):
    with torch.no_grad():
        logits = model(batch)
        x = batch.x.reshape(-1,total_num).clip(min=precision,max=1-precision)
        phi = F.softmax(logits[:,:main_num], dim=1)
        phi = torch.cat((phi, F.sigmoid(logits[:,main_num:])),dim=1)
        phi = phi.clip(min=precision,max=1-precision)
        gradient = 0
        for i in range(sample_num):
            index = Categorical(probs = phi[:,:main_num]).sample().reshape(-1,1)
            sample = torch.zeros_like(phi[:,:main_num])
            sample.scatter_(index=index, dim=1, value=1)
            sample = torch.cat((sample, Bernoulli(probs = phi[:,main_num:]).sample()),dim=1)
            
            sample = sample.unsqueeze(1).repeat(1,sample.shape[-1],1)
            flip = torch.diag(torch.ones(sample.shape[-1])).unsqueeze(0).cuda()
            flip = (flip - sample).abs()
            n1 = neg_indicator(sample @ formula + bias)
            n2 =  neg_indicator(flip @ formula + bias)
            s1 = torch.zeros_like(sample)
            tmp_s1 = x.unsqueeze(1).repeat(1,x.shape[-1],1)[sample!=0]
            s1[sample!=0] = (tmp_s1/(1-tmp_s1)).log()
            s2 = torch.zeros_like(flip)
            tmp_s2 = x.unsqueeze(1).repeat(1,x.shape[-1],1)[flip!=0]
            s2[flip!=0] = (tmp_s2/(1-tmp_s2)).log()
            s1 = torch.cat((s1, n1 * model.w.exp()), dim=-1)
            s2 = torch.cat((s2, n2 * model.w.exp()), dim=-1)
            p1 = (s1 * new_formula.abs()).sum(-1,keepdims=True)
            p2 = (s2 * new_formula.abs()).sum(-1,keepdims=True)
            p1, p2 = F.softmax(torch.cat((p1,p2),dim=-1),dim=-1).chunk(2,-1)
            gradient += (n1 - p1 * n1 - p2 * n2).sum(1).mean(0)/sample_num
        model.w.data += gradient * lr * model.w.exp()
           
def main():
    args = parse_args()
    train_dataset = get_gcn_dataset(dataset=args.dataset, noise_sd=args.noise_sd, split='train')
    formula, bias = train_dataset.get_w()
    total_num = train_dataset.total_num 
    main_num = train_dataset.main_num 
    sample_num = args.sample_num  

    new_formula = torch.cat((torch.eye(formula.shape[0]), formula), dim=1).cuda()
    model = GCN(total_num, formula.shape[1]).cuda()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model.train()
    for epoch in range(args.training_epoch):
        total, correct = 0, 0
        for batch in train_loader:
            batch = batch.cuda()
            optimizer.zero_grad()
            loss = E_step(model, batch, sample_num)
            loss.backward()
            optimizer.step()
            M_step(model, batch, scheduler.get_last_lr()[0], sample_num)

            output = model(batch)
            label = batch.y.view(-1, total_num).cuda()
            total += output.shape[0]
            correct += (output[:, :main_num].argmax(1) == label[:, :main_num].argmax(1)).sum().item()

        scheduler.step()
        print(f"GCN Training - epoch: {epoch}, acc: {correct / total:.4f}")

    # Save additional metadata with the model
    save_dict = {
        'state_dict': model.state_dict(),
        'input_dim': total_num,
        'formula_dim': formula.shape[1]
    }
    torch.save(save_dict, f'gcn_models/gcn_{args.dataset}_noise_sd{args.noise_sd:.2f}_w{args.w}.pt')

if __name__ == '__main__':
    main()

