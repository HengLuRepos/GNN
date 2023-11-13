import torch
from config import Config
from lightGCN import LightGCN, Dataset
import numpy as np
cfg = Config()
yelp2018 = Dataset("./data/yelp2018/")
agent = LightGCN(yelp2018, cfg)
optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
batch_size = cfg.user_sample * (cfg.neg_ratio + 1) * cfg.pos_number
num_batch = yelp2018.userItemMatrix.count_nonzero() // batch_size + 1
for ep in range(cfg.epoch):
    for batch in range(num_batch):
        users_index = np.random.randint(low=0, high=yelp2018.num_user, size=cfg.user_sample)
        pos_items = []
        neg_items = []
        for index, user_index in enumerate(users_index):
            all_connected_items = yelp2018.userItemMatrix[user_index].nonzero()[1]
            pos_item_user = np.random.choice(all_connected_items, size=cfg.pos_number)
            pos_items.append(pos_item_user)
            neg_item_user = []
            while len(neg_item_user) < cfg.pos_number * cfg.neg_ratio:
                item = np.random.choice(yelp2018.num_item)
                if item not in all_connected_items:
                    neg_item_user.append(item)
            neg_items.append(neg_item_user)

        loss = agent.bpr_loss(users_index, pos_items, neg_items)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Ep {ep} batch {batch}: Loss {loss.item():.4f}")
