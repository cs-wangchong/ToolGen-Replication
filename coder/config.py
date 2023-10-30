
class Config:
    def __init__(
        self,
        model_name="salesforce/codet5",
        max_len=512,
        train_batch_size=1,
        valid_batch_size=1,
        epochs=2,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup_steps=0.1,
        max_grad_norm=1.0,
        train_sampling=None,
        valid_sampling=None,
        save_dir=None,
        log_step=500,
        valid_step=1000,
        cand_num=10,
        best_k=3,
        patience=3,
        beam_size=5,
        repetition_penalty=2.5,
        device="cuda"
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.train_batch_size=train_batch_size
        self.valid_batch_size=valid_batch_size
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.adam_epsilon=adam_epsilon
        self.weight_decay=weight_decay
        self.warmup_steps=warmup_steps
        self.max_grad_norm=max_grad_norm
        self.train_sampling=train_sampling
        self.valid_sampling=valid_sampling
        self.save_dir=save_dir
        self.log_step=log_step
        self.valid_step = valid_step
        self.cand_num=cand_num
        self.best_k = best_k
        self.patience = patience
        self.beam_size = beam_size
        self.repetition_penalty = repetition_penalty
        self.device = device

    def to_json(self):
        return {k:v for k, v in self.__dict__.items() if hasattr(self, k)}

    @classmethod
    def from_json(cls, json_data):
        config = Config()
        for k, v in json_data.items():
            setattr(config, k, v)
        return config
