import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC(id_ec, ec_id, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader

def normalize_id(pid):
    return pid.split('_')[0]

def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion, id_ec):
    model.train()
    total_loss = 0.
    start_time = time.time()

    log_path = f"./triplet_logs/{args.model_name}_triplets.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()

        # Unpack data
        anchor, positive, negative, anchor_id, pos_id, neg_id = data

        anchor_out = model(anchor.to(device=device, dtype=dtype))
        positive_out = model(positive.to(device=device, dtype=dtype))
        negative_out = model(negative.to(device=device, dtype=dtype))

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print first 3 triplets for inspection
        # print(f"\nEpoch {epoch}, Batch {batch} — Sample triplets:")
        for i in range(min(3, len(anchor_id))):
            a_id = normalize_id(anchor_id[i])
            p_id = normalize_id(pos_id[i])
            n_id = normalize_id(neg_id[i])
            # print(f"  Anchor: {anchor_id[i]} ({id_ec[a_id]}), "
            #     f"Positive: {pos_id[i]} ({id_ec[p_id]}), "
            #     f"Negative: {neg_id[i]} ({id_ec[n_id]})")
        # for i in range(min(3, len(anchor_id))):
        #     print(f"  Anchor: {anchor_id[i]} ({id_ec[anchor_id[i]]}), "
        #           f"Positive: {pos_id[i]} ({id_ec[pos_id[i]]}), "
        #           f"Negative: {neg_id[i]} ({id_ec[neg_id[i]]})")

        # Save triplets and their EC numbers to CSV
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['epoch', 'batch', 
                                 'anchor_id', 'anchor_ec',
                                 'positive_id', 'positive_ec',
                                 'negative_id', 'negative_ec'])
                write_header = False
            for a, p, n in zip(anchor_id, pos_id, neg_id):
                a_clean = normalize_id(a)
                p_clean = normalize_id(p)
                n_clean = normalize_id(n)
                writer.writerow([epoch, batch,
                                a, id_ec[a_clean],
                                p, id_ec[p_clean],
                                n, id_ec[n_clean]])
            # for a, p, n in zip(anchor_id, pos_id, neg_id):
            #     writer.writerow([epoch, batch,
            #                      a, id_ec[a],
            #                      p, id_ec[p],
            #                      n, id_ec[n]])

        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {total_loss:5.2f}')
            start_time = time.time()

    return total_loss / (batch + 1)


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
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
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
    # remove tmp save weights
    #os.remove('./data/model/' + model_name + '.pth')
    #os.remove('./data/model/' + model_name + '_' + str(epoch) + '.pth')
    # save final weights
    torch.save(model.state_dict(), './data/model/' + model_name + '.pth')


if __name__ == '__main__':
    main()
