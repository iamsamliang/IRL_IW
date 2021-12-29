from abc import ABC, abstractclassmethod
from msdm.algorithms.entregpolicyiteration import entropy_regularized_policy_iteration
from torch.utils.data.dataloader import DataLoader
from dataset import TrajectoryDataset, FeaturesDataset
import torch
import numpy as np
import frozendict


class Algorithm(ABC):
    @abstractclassmethod
    def fit(self):
        pass


class MaxLikelihoodIRL(Algorithm):
    @classmethod
    def fit(self, mdp, trajectories, batch_size=64, epochs=50, lr=1):
        '''
        Parameters:
          mdp: a representation of the Markov Decision Process (x-coords, y-coords, color of states, and transition probabilities)
          feature_matrix: S | 1 x A | 1 x S x M where the arr[i, j, k, :] = the features of taking action j in state i and transitioning to state k
          step_cost_matrix: S x A | 1 x S | 1 - action at every state has a step cost, except the terminal state
          trajectories: array of trajectories which are dictionaries containing the state and action trajectory
          batch_size: size of a batch of data for training
          epochs: number of epochs for learning the reward function
          lr: learning rates for each iteration (scalar or array of length epochs)
        Output:
          r_weights: the learned reward weights
          policy: policy associated with the final r_weights
        '''

        trajs_dataset = TrajectoryDataset(trajectories)
        trajs_dataloader = DataLoader(
            trajs_dataset, batch_size=batch_size, shuffle=True)

        feature_matrix = self.__getfeaturematrix__(mdp)
        step_cost_matrix = self.__getstepcostmatrix__(mdp)

        # randomize weights
        r_weights = torch.tensor(np.random.randn(feature_matrix.shape[-1]))

        # try using the true reward weights to debug - Checked. Has basically the same rewards
        # r_weights = torch.tensor([float(mdp_params['feature_rewards'][f]) for f in features]).double()

        r_weights.requires_grad = True
        discount_rate = torch.tensor(mdp.discount_rate)
        transition_matrix = torch.tensor(mdp.transition_matrix)

        # parameters for policy algorithm
        entropy_weight = 1
        planning_iters = 10

        optimizer = torch.optim.SGD([r_weights], lr)

        size = len(trajs_dataloader.dataset)

        for _ in range(epochs):
            for batch, (state_trajectories, action_trajectories, trajectories_weights) in enumerate(trajs_dataloader):
                optimizer.zero_grad()

                # build the reward matrix of the entire MDP using current reward weights

                # We do not have the reward weights from MDP bc the reward weights are what we are optimizing. Thus, use r_weights
                feature_reward_matrix = torch.einsum(
                    "sanf,f->san",
                    feature_matrix,
                    r_weights
                )

                # construct the final reward function by taking into account each action's step cost
                reward_matrix = feature_reward_matrix + step_cost_matrix

                # this should be |S| x 1 x |S| since its the same for all actions
                assert tuple(reward_matrix.shape) == (
                    len(mdp.state_list), 1, len(mdp.state_list))

                # anything leading to the terminal state ({-1, -1}) has zero reward
                terminal_index = mdp.state_index.get(
                    frozendict({'x': -1, 'y': -1}))
                reward_matrix[:, :, terminal_index] = 0

                # compute the optimal policy for the current reward function
                policy = entropy_regularized_policy_iteration(
                    transition_matrix, reward_matrix, discount_rate, entropy_weight, planning_iters).policy
                loss = torch.tensor(0.0)

                for traj_index in range(len(state_trajectories)):
                    state_traj = state_trajectories[traj_index]
                    action_traj = action_trajectories[traj_index]
                    trajectory_weight = trajectories_weights[traj_index]
                    for index in range(len(state_traj)):
                        state = state_traj[index]
                        action = action_traj[index]
                        state_index = mdp.state_index.get(state)
                        action_index = mdp.action_index.get(action)
                        loss -= torch.log(policy[state_index,
                                          action_index]) * trajectory_weight
                loss.backward()
                optimizer.step()

                if batch % 40000 == 0:
                    loss, current = loss.item(), batch * len(state_trajectories)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # compute policy from learned weights
        feature_reward_matrix = torch.einsum(
            "sanf,f->san",
            feature_matrix,
            r_weights
        )

        reward_matrix = feature_reward_matrix + step_cost_matrix
        terminal_index = mdp.state_index.get(
            frozendict({'x': -1, 'y': -1}))
        reward_matrix[:, :, terminal_index] = 0
        policy = entropy_regularized_policy_iteration(
            transition_matrix, reward_matrix, discount_rate, entropy_weight, planning_iters).policy

        return r_weights, policy

    def __getfeaturematrix__(self, mdp):
        features = sorted(mdp.feature_locations.keys())
        features = [f for f in features if f not in 'gs']

        # each state is a one-hot vector of its color
        state_feature_matrix = torch.zeros(len(mdp.state_list), len(features))
        states = mdp.state_index

        for state, state_index in states.items():
            f = mdp.location_features.get(state, '.')
            if f in features:
                feature_index = features.index(f)
            else:
                continue
            state_feature_matrix[state_index, feature_index] = 1

        # shape is 1 (state) x 1 (action) x num_states x num_features bc the features only depend on the next_state so for any state and any action, if it ends in next_state i, it should have the features corresponding to next_state i
        state_action_nextstate_feature_matrix = state_feature_matrix[None, None, :]
        return state_action_nextstate_feature_matrix.double()

    def __getstepcostmatrix__(self, mdp):
        # action at every state has a step cost, except the terminal state
        state_step_cost_matrix = torch.tensor(
            mdp.step_cost) * mdp.nonterminal_state_vec

        # shape is num_states (state) x 1 (action) x 1 (next_state) bc the step cost is only dependent on the action you take in the current state. Since it is the same for all actions (and what next_state you end up at), we only need to encode it for the 26 states
        step_cost_matrix = state_step_cost_matrix[:, None, None]
        return step_cost_matrix


class ImitationLearning(Algorithm):
    @classmethod
    def fit(self, mdp, trajectories, model, loss_fn, optimizer, scheduler=None, batch_size=128, epochs=50):
        '''
        Parameters:
          mdp: a representation of the Markov Decision Process
          trajectories: array of trajectories which are dictionaries containing the state and action trajectory
          model: PyTorch neural network
          loss_fn: PyTorch loss function
          optimizer: PyTorch optimizer
          scheduler: PyTorch scheduler (varies learning rate at specific epochs)
          batch_size: size of a batch of data for training
          epochs: number of epochs for learning the reward function
        Output:
          model: trained PyTorch model
        '''

        features_dataset = FeaturesDataset(
            mdp, trajectories, transform=self.__transform__)
        features_dataloader = DataLoader(
            features_dataset, batch_size=batch_size, shuffle=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.__train__(features_dataloader, model,
                           loss_fn, optimizer, device)
            print()
            if scheduler is not None:
                scheduler.step()
        print("Done!")

        return model

    def __transform__(self, sample):
        return torch.from_numpy(sample)

    def __train__(self, dataloader, model, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 40000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
