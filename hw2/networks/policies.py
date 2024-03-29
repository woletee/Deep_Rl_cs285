import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import torch
from torch import distributions
import torch.nn as nn
from cs285.infrastructure import pytorch_util as ptu
from torch.nn import nn 
#this is concerned with determinng which actions to take being in a specified state.
#concerned with the policy of the agent which maps states to actions
#the output is basically action distiribution 
class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain(
                [self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # DONE: implement get_action
        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs)
        # Using rsample to allow gradients to pass through the sample

        if self.discrete:
            action = action_distribution.sample()
        else:
            action = action_distribution.rsample()
        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # DONE: define the forward pass for a policy with a discrete action space.
            action_distribution = distributions.Categorical(
                logits=self.logits_net(obs)
            )
        else:
            # DONE: define the forward pass for a policy with a continuous action space.
            action_distribution = distributions.Normal(
                loc=self.mean_net(obs),
                scale=torch.exp(self.logstd)
            )
        return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """
        Implements the policy gradient actor update. 
        This method performs a policy update using given observations, actions,
        and computed advantages.

        Parameters:
        - obs (np.ndarray): A batch of observations used for the policy update.
        - actions (np.ndarray): Corresponding actions taken in the observed states.
        - advantages (np.ndarray): The advantages associated with the pairs of 
                                   observations and actions.

        Returns:
        - dict: A dictionary containing the policy loss after the update.

        Note:
        This method expects numpy arrays as inputs, converts them to PyTorch tensors,
        performs the update, and then converts the results back to numpy for consistency.
        """
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # DONE: implement the policy gradient actor update.
        action_distribution = self.forward(obs)
        if self.discrete:
            log_prob = action_distribution.log_prob(actions)
        else:
            # For continuous action spaces, actions are typically multidimensional,
            # so we sum the log prob across dimensions
            log_prob = action_distribution.log_prob(actions).sum(dim=-1)

        loss = -(log_prob * advantages).mean()  # Negative for gradient ascent
###may be sum ()
        # Backward pass on the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
