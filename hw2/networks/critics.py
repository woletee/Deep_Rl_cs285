
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu

#this is used to estimate the value of being in a given state according to the current policy 
#value refers to the expected rewards that will be gained being in a given state.
#it will output a single scaler value which indeicates the value of being in a specific state.
#the main objective is to minimize the difference between the actual states values and the predicted states values
#this critic assumes that the agent has interacted with the environment follwing some policy and then has the actual values of being in that states
#using the actual vlaues and the one predicted by the critic , this class willcompute the loss.
class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # DONE: implement the forward pass of the critic network
        return self.network(obs).squeeze()

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # DONE: update the critic using the observations and q_values

        # Compute the predicted values
        pred_values = self.forward(obs)
        # Compute the loss
        loss = F.mse_loss(pred_values, q_values)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }
