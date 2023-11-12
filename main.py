import torch
from config import Config
from lightGCN import LightGCN, Dataset
import numpy as np
cfg = Config()
yelp2018 = Dataset("./data/yelp2018/")
agent = LightGCN(yelp2018, cfg)
optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
num_batch = yelp2018.num_user // cfg.batch_size + 1
for ep in range(cfg.epoch):
    for batch in range(num_batch):
        user_index = np.random.randint(low=0, high=yelp2018.num_user, size=cfg.batch_size)
        loss = agent.bpr_loss(user_index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Ep {ep} batch {batch}: Loss {loss.item():.4f}")
