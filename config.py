class Config:
    def __init__(self) -> None:
        #for lightGCN
        self.layers = 3
        self.embedding_dim = 64
        self.weight_decay = 1e-4
        self.epoch = 1000
        self.neg_ratio = 4
        self.user_sample = 500
        self.pos_number = 1
        self.lr = 1e-3

        self.noise_norm = 1e-1
        self.lam = 0.5