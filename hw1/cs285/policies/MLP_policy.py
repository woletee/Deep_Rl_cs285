"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
      
        observation = observation.float().to(ptu.device) #ensuring the observation is torch.floatTensor
        mean_action = self.mean_net(observation)  #the observation is then passed through the mean_net function which has been defined before to compute the mean of the action distribution 
        std = torch.exp(self.logstd) #here we will compute the standard deviation of the and here we it is calculated as log to maintain that the standard deviation is always posetive.
        action_distribution = distributions.Normal(mean_action, std)  #for each action using the calculated mean and then standard deviation a normal distirbution is consturcted.
        sampled_action = action_distribution.rsample() #from the calculated normal distiribution an action is sampled.
        return sampled_action #finally we will return the sampled action
#the mean and the standard deviations calculated are basiclly the parameters for the probability distiribution (gaussiab distirbuiotns )
#mean of the action distirbuiton is the expected value of the action given the current state
#standard deviation is about the how much variance there is in the actions that the policy might take 
    def update(self, observations, actions):
    #this method will update the parameters of the policy 
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # Predict actions for the given observations
        predicted_actions = self.forward(observations)

        # Compute the Mean Squared Error loss between the predicted and target actions
        loss = F.mse_loss(predicted_actions, actions)

        # Zero the gradients before backpropagation
        self.optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Perform a step of optimization
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
this function will retunr the actions that should be taken given an obaservation 
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(
            observation, dtype=torch.float32).to(ptu.device)
        with torch.no_grad(): #here we have disabled the gradiant computation this is because here we are not training thus we donot need to compute the loss 
            action = ptu.to_numpy(self.forward(observation)) #here we generate the action from the polciy using the function forward 
        return action  #this will return the actions recommended by the forwrad function of the polciy 
#the get_action function converts the pytorch tensor actions into num_array which can be ysed latter for the neural network computation 
