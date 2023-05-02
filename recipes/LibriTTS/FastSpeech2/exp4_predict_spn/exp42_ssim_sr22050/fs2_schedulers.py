"""
Schedulers for updating hyperparameters (such as learning rate).
Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Loren Lugosch 2020
"""

import math
import torch
import logging

from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)

@checkpoints.register_checkpoint_hooks
class NoamIntervalScheduler:
    """A combination of Noam Scheduler and Interval Scheduler.
    The scheduler behaves as a Noam Scheduler, and anneals the learning rate
    at disigned steps with designed decays.
    Note: this scheduler anneals the lr at each update of the model's weight,
    and n_steps must be saved for restarting.
    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm-up steps.
    anneal_steps: list
        Pre-designed steps where the learning rate is to be annealed.
    anneal_rates: list
        Pre-designed decay rate for each anneal step.
    model_size : int
        size of transformer embed_dim. It is used to scale the maximum learning rate value reached
        by the scheduler. It is divided by model_size ** (0.5).
        If not specified the maximum learning rate value is instead multiplied by warmup_steps ** (0.5).
    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler = NoamIntervalScheduler(
    ...    lr_initial=optim.param_groups[0]["lr"],
    ...    n_warmup_steps=3,
    ...    anneal_steps=[6, 9],
    ...    anneal_rates=[0.5, 0.1],
    ... )
    >>> for _ in range(10):
    ...     curr_lr,next_lr=scheduler(optim)
    ...     print(optim.param_groups[0]["lr"])
    0.3333333333333333
    0.6666666666666666
    0.9999999999999999
    0.8660254037844386
    0.7745966692414833
    0.7071067811865475
    0.3273268353539886
    0.3061862178478973
    0.28867513459481287
    0.027386127875258306
    """

    def __init__(
        self,
        lr_initial,
        n_warmup_steps,
        anneal_steps,
        anneal_rates,
        model_size=None,
    ):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        self.anneal_steps = anneal_steps
        self.anneal_rates = anneal_rates
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.
        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        lr_scale = self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )
        for i in range(len(self.anneal_steps)):
            if self.n_steps > self.anneal_steps[i]:
                lr_scale = lr_scale * self.anneal_rates[i]
        return lr_scale

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]