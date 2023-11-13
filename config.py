class Config:
    def __init__(self) -> None:
        self.layers = 3
        self.embedding_dim = 64
        self.weight_decay = 1e-4
        self.epoch = 1000
        self.neg_ratio = 4
        self.user_sample = 200
        self.pos_number = 5
        self.lr = 3e-3