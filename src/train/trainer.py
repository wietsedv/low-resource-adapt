from transformers import (Trainer as HFTrainer, AdamW,
                          get_linear_schedule_with_warmup)


class CustomTrainer(HFTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        named_params = [(n, p) for n, p in self.model.named_parameters()
                        if p.requires_grad]

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in named_params
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in named_params
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps)
