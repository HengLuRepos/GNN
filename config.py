class Config:
    def __init__(self) -> None:
        self.layers = 3
        self.embedding_dim = 64
        self.weight_decay = 1e-4
        self.epoch = 1000
        self.batch_size = 5000
        self.lr = 3e-3