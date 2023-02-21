import torch
from torch import nn

from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim


def init_method_1(model):
    # model.weight.data.uniform_(-1, 1)
    # model.bias.data.uniform_(-1, 1)

    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.model_f = ptu.build_mlp(input_size=self.ob_dim,
                                     output_size=self.output_size,
                                     n_layers=self.n_layers,
                                     size=self.size,
                                     init_method=init_method_1)

        self.model_f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                         output_size=self.output_size,
                                         n_layers=self.n_layers,
                                         size=self.size,
                                         init_method=init_method_2)
        self.model_f.to(ptu.device)
        self.model_f_hat.to(ptu.device)

        self.optimizer = self.optimizer_spec.constructor(
            self.model_f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        pred_f = self.model_f(ob_no).detach()
        pred_f_hat = self.model_f_hat(ob_no)

        prediction_error = (pred_f - pred_f_hat).square().mean(-1)

        return prediction_error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)

        prediction_error = self(ob_no)
        prediction_error = prediction_error.mean()

        self.optimizer.zero_grad()
        prediction_error.backward()
        self.optimizer.step()

        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(prediction_error),
        }
