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
        #simGCL
        self.noise_norm = 1e-1
        self.lam = 0.5
        self.tau = 0.2
        #JGCF
        self.a = 1.0
        self.b = 1.0
        self.alpha = 0.2