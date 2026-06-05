""" Base learner class."""

import torch


class BaseLearner:
    """Base learner class.

    Parameters
    ----------
    model: pytorch model
        Pytorch model to be trained.
    adam_params: dict, optional
        Dictionary of Adam parameters.
    lr_schedule: dict, optional
        Dictionary of learning rate scheduler parameters.

    """

    def __init__(
        self,
        model,
        adam_params=None,
        lr_schedule=None,
    ):
        self.model = model

        # set up optimizer
        if adam_params is None:
            # use default
            adam_params = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_params)

        # set up learning-rate scheduler
        self.lr_decay = lr_schedule is not None
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **lr_schedule
            )

    def lr_scheduler_step(self, metric):
        """Update learning rate."""
        if self.lr_decay:
            self.scheduler.step(metric)

    def train_step(self, batch):
        """Perform one training step on a batch (forward + backward)."""
        self.model.train()
        self.optimizer.zero_grad()
        loss, predictions, true_loss = self.forward_pass(batch)
        loss.backward()
        self.optimizer.step()
        return loss, predictions, true_loss

    def forward_pass(self, inputs, labels, *args, **kwargs):
        return None, None
