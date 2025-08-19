import torch
import time
import os
import pickle
import random
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.losses import SupConHardLoss
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
import csv


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=1500)
    parser.add_argument('-n', '--model_name', type=str, default='split10_supconH')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    # ------------  SupCon-Hard specific  ------------ #
    parser.add_argument('-T', '--temp', type=float, default=0.1)
    parser.add_argument('--n_pos', type=int, default=9)
    parser.add_argument('--n_neg', type=int, default=30)
    # ------------------------------------------- #
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=256)
    parser.add_argument('--adaptive_rate', type=int, default=60)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNeg_dataset_with_mine_EC(
        id_ec, ec_id, negative, args.n_pos, args.n_neg)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def normalize_id(pid):
    return pid.split('_')[0]

def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion, id_ec):
    model.train()
    total_loss = 0.
    start_time = time.time()
    log_path = f"./triplet_logs/{args.model_name}_supconH_pairs_100_epochs.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    
    def flatten_to_str(x):
        while isinstance(x, (tuple, list)):
            x = x[0]
        return str(x)
    
    for batch, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        data, ids, ecs = batch_data
        model_emb = model(data.to(device=device, dtype=dtype))
        loss = criterion(model_emb, args.temp, args.n_pos)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Logging - extract anchor, positives, negatives from batch
        anchor_id = flatten_to_str(ids[0])
        anchor_ec = flatten_to_str(ecs[0])
        pos_ids = ids[1:1+args.n_pos]
        pos_ecs = ecs[1:1+args.n_pos]
        neg_ids = ids[1+args.n_pos:]
        neg_ecs = ecs[1+args.n_pos:]

        # Save pairs and their EC numbers to CSV
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['epoch', 'batch', 
                                'anchor_id', 'anchor_ec',
                                'positive_ids', 'positive_ecs',
                                'negative_ids', 'negative_ecs'])
                write_header = False
            writer.writerow([epoch, batch,
                            anchor_id, anchor_ec,
                            '|'.join(flatten_to_str(x) for x in pos_ids), '|'.join(flatten_to_str(x) for x in pos_ecs),
                            '|'.join(flatten_to_str(x) for x in neg_ids), '|'.join(flatten_to_str(x) for x in neg_ecs)])
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000  
            cur_loss = total_loss  
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)

class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):
    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = [ec for ec in ec_id if '-' not in ec and len(ec.split('.')) == 4]
        self.mine_neg = mine_neg

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load(f'./data/esm_data/{anchor}.pt')).unsqueeze(0)
        # Positives: same EC, not anchor
        pos_candidates = [x for x in self.ec_id[anchor_ec] if x != anchor]
        pos_ids, pos_tensors = [], []
        for _ in range(self.n_pos):
            pos = random.choice(pos_candidates) if pos_candidates else anchor
            pos_ids.append(pos)
            p = format_esm(torch.load(f'./data/esm_data/{pos}.pt')).unsqueeze(0)
            pos_tensors.append(p)
        # Negatives: same first 3 EC, different 4th
        ec_prefix = '.'.join(anchor_ec.split('.')[:3])
        neg_ecs = [ec for ec in self.ec_id if ec.startswith(ec_prefix + '.') and ec != anchor_ec]
        neg_ids, neg_tensors = [], []
        neg_ec_list = []
        for _ in range(self.n_neg):
            if neg_ecs:
                neg_ec = random.choice(neg_ecs)
                neg_seq = random.choice(self.ec_id[neg_ec])
                neg_ec_list.append(neg_ec)
            else:
                neg_seq = anchor
                neg_ec_list.append(anchor_ec)
            neg_ids.append(neg_seq)
            n = format_esm(torch.load(f'./data/esm_data/{neg_seq}.pt')).unsqueeze(0)
            neg_tensors.append(n)
        data = [a] + pos_tensors + neg_tensors
        ids = [anchor] + pos_ids + neg_ids
        ecs = [anchor_ec] + [anchor_ec]*len(pos_ids) + neg_ec_list
        return torch.cat(data), ids, ecs

def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
 
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = SupConHardLoss
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./data/model/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion, id_ec)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print('-' * 75)
    # # remove tmp save weights
    # os.remove('./data/model/' + model_name + '.pth')
    # os.remove('./data/model/' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), './data/model/' + model_name + '.pth')


if __name__ == '__main__':
    main()
