import torch
import random
from .utils import format_esm
from tqdm import tqdm

print("data loader")

def find_first_non_zero_distance(data):
    for index, (_, distance) in enumerate(data):
        if distance != 0:
            return index
    return None

def mine_hard_negative(dist_map, knn=10):
    ecs = list(dist_map.keys())  # full EC4 keys
    negative = {}
    print("Mining hard negatives at EC4 level:")
    for _, target in tqdm(enumerate(ecs), total=len(ecs)):
        sorted_orders = sorted(dist_map[target].items(), key=lambda x: x[1])
        neg_start_idx = find_first_non_zero_distance(sorted_orders)
        closest_negatives = sorted_orders[neg_start_idx:neg_start_idx + knn]
        weights = [1 / i[1] for i in closest_negatives]
        normalized = [w / sum(weights) for w in weights]
        negative[target] = {
            'weights': normalized,
            'negative': [i[0] for i in closest_negatives]
        }
    return negative

def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec_list = id_ec[anchor]
    anchor_ecs = [ec for ec in anchor_ec_list if len(ec.split('.')) == 4]

    if not anchor_ecs:
        return anchor  # fallback if no EC4

    # Pick one EC4 to work with
    anchor_ec = random.choice(anchor_ecs)
    prefix = '.'.join(anchor_ec.split('.')[:3])  # e.g., "7.1.1"

    # Find candidate ECs with same first 3 parts but different 4th
    candidate_ecs = [ec for ec in mine_neg.get(anchor_ec, {}).get('negative', [])
                     if ec.startswith(prefix) and ec != anchor_ec]

    if not candidate_ecs:
        return anchor  # fallback

    # Sample based on weights
    weights = mine_neg[anchor_ec]['weights']
    idx_map = {ec: i for i, ec in enumerate(mine_neg[anchor_ec]['negative'])}
    filtered_ecs = [(ec, weights[idx_map[ec]]) for ec in candidate_ecs if ec in idx_map]

    if not filtered_ecs:
        return anchor  # fallback

    ec_list, ec_weights = zip(*filtered_ecs)
    result_ec = random.choices(ec_list, weights=ec_weights, k=1)[0]

    # Choose protein ID from result EC
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    # print("random positive", pos_ec, pos, id)
    return pos

# def mine_negative(anchor, id_ec, ec_id, mine_neg):
#     anchor_ec_list = id_ec[anchor]
#     anchor_ec = anchor_ec_list[0] if anchor_ec_list else None
#     if anchor_ec is None or anchor_ec not in mine_neg:
#         # fallback: just return anchor itself
#         return anchor

#     candidates = mine_neg[anchor_ec]['negative']
#     weights = mine_neg[anchor_ec]['weights']

#     # Pick target_ec from candidates ensuring it is in ec_id and not same as anchor_ec
#     target_ec = None
#     max_attempts = 10
#     attempts = 0
#     while attempts < max_attempts:
#         target_ec = random.choices(candidates, weights=weights, k=1)[0]
#         if target_ec != anchor_ec and target_ec in ec_id:
#             break
#         attempts += 1
#     else:
#         # If no valid target_ec found, fallback
#         return anchor

#     return random.choice(list(ec_id[target_ec]))

# def random_positive(anchor, id_ec, ec_id):
#     ec4 = id_ec[anchor][0]  # Fix: get first EC string from list
#     pos = anchor
#     if len(ec_id[ec4]) == 1:
#         return pos
#     while pos == anchor:
#         pos = random.choice(list(ec_id[ec4]))
#     return pos

class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):
    def __init__(self, id_ec, ec_id, mine_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.mine_neg = mine_neg
        self.full_list = [ec for ec in ec_id if '-' not in ec and len(ec.split('.')) == 4]
        self.triplets = []  # optional for logging/debugging

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        a = torch.load(f'./data/esm_data/{anchor}.pt')
        p = torch.load(f'./data/esm_data/{pos}.pt')
        n = torch.load(f'./data/esm_data/{neg}.pt')
        # print(format_esm(a), format_esm(p), format_esm(n), anchor, pos, neg)
        return format_esm(a), format_esm(p), format_esm(n), anchor, pos, neg


# import torch
# import random
# from .utils import format_esm
# from tqdm import tqdm

# def find_first_non_zero_distance(data):
#     for index, (name, distance) in enumerate(data):
#         if distance != 0:
#             return index
#     return None 

# def mine_hard_negative(dist_map, knn=10):
#     #print("The number of unique EC numbers: ", len(dist_map.keys()))
#     ecs = list(dist_map.keys())
#     negative = {}
#     print("Mining hard negatives:")
#     for _, target in tqdm(enumerate(ecs), total=len(ecs)):
#         sorted_orders = sorted(dist_map[target].items(), key=lambda x: x[1], reverse=False)
#         assert sorted_orders != None, "all clusters have zero distances!"
#         neg_ecs_start_index = find_first_non_zero_distance(sorted_orders)
#         closest_negatives = sorted_orders[neg_ecs_start_index:neg_ecs_start_index + knn]
#         freq = [1/i[1] for i in closest_negatives]
#         neg_ecs = [i[0] for i in closest_negatives]        
#         normalized_freq = [i/sum(freq) for i in freq]
#         negative[target] = {
#             'weights': normalized_freq,
#             'negative': neg_ecs
#         }
#     return negative


# def mine_negative(anchor, id_ec, ec_id, mine_neg):
#     anchor_ec = id_ec[anchor]
#     pos_ec = random.choice(anchor_ec)
#     neg_ec = mine_neg[pos_ec]['negative']
#     weights = mine_neg[pos_ec]['weights']
#     result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
#     while result_ec in anchor_ec:
#         result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
#     neg_id = random.choice(ec_id[result_ec])
#     return neg_id


# def random_positive(id, id_ec, ec_id):
#     pos_ec = random.choice(id_ec[id])
#     pos = id
#     if len(ec_id[pos_ec]) == 1:
#         return pos + '_' + str(random.randint(0, 9))
#     while pos == id:
#         pos = random.choice(ec_id[pos_ec])
#     return pos


# class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

#     def __init__(self, id_ec, ec_id, mine_neg):
#         self.id_ec = id_ec
#         self.ec_id = ec_id
#         self.full_list = []
#         self.mine_neg = mine_neg
#         for ec in ec_id.keys():
#             if '-' not in ec:
#                 self.full_list.append(ec)

#     def __len__(self):
#         return len(self.full_list)

#     def __getitem__(self, index):
#         anchor_ec = self.full_list[index]
#         anchor = random.choice(self.ec_id[anchor_ec])
#         pos = random_positive(anchor, self.id_ec, self.ec_id)
#         neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
#         a = torch.load('./data/esm_data/' + anchor + '.pt')
#         p = torch.load('./data/esm_data/' + pos + '.pt')
#         n = torch.load('./data/esm_data/' + neg + '.pt')
#         return format_esm(a), format_esm(p), format_esm(n)


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        ids = [anchor]
        ecs = [anchor_ec]
        
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
            ids.append(pos)
            ecs.append(self.id_ec[pos])
            
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
            ids.append(neg)
            ecs.append(self.id_ec[neg])
            
        return torch.cat(data), ids, ecs
