from abc import ABC, abstractmethod
from msdm.algorithms.entregpolicyiteration import entropy_regularized_policy_iteration
from torch.utils.data.dataloader import DataLoader
from dataset import TrajectoryDataset, FeaturesDataset
import torch
import numpy as np
from frozendict import frozendict


class Algorithm(ABC):
    @classmethod
    @abstractmethod
    def learn(self, dataset: TrajectoryDataset):
        '''Takes in a dataset containing formatted trajectories, performs optimization on parameters determined by the algorithm, and returns an object containing the learned parameters
        '''


class MaxLikelihoodIRL(Algorithm):
    def __init__(self, mdp, featurizer, fixed_reward, batch_size=128, epochs=50, lr=0.1, weight_decay=0, momentum=0, entropy_weight=1, planning_iters=10):
        '''
        Parameters:
          mdp: a representation of the Markov Decision Process
          featurizer: func(state, action, next_state) -> dictionary w/ features as keys and float|int associated with each feature as values; unspecified features will have a value of 0
          fixed_reward: func(state, action, next_state) -> float
          batch_size: size of a batch of data for training
          epochs: number of epochs for learning the reward function
          lr: learning rates for each iteration (scalar or array of length epochs)
          entropy_weight: for entropy policy (see msdm entropy_regularized_policy_iteration)
          planning_iters: for entropy policy (see msdm entropy_regularized_policy_iteration)
        Internal Outputs:
          feature_matrix: S | 1 x A | 1 x S x F where the arr[i, j, k, :] = the features of taking action j in state i and transitioning to state k, F = number of features
          fixed_reward_matrix: S x A | 1 x S | 1 - action at every state has a fixed reward
        '''
        self.mdp = mdp
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.entropy_weight = entropy_weight
        self.planning_iters = planning_iters

        num_states = len(self.mdp.state_list)
        num_actions = len(self.mdp.action_list)

        # taking an action in a state and transitioning to another state has a fixed reward
        fixed_reward_matrix = np.zeros((num_states, num_actions, num_states))
        states = mdp.state_index
        actions = mdp.action_index
        # generate the appropriate fixed reward from user-defined fixed reward function
        for state, state_index in states.items():
            for action, action_index in actions.items():
                for next_state in mdp.next_state_dist(state, action).support:
                    next_state_index = states.get(next_state)
                    fixed_reward_matrix[state_index, action_index, next_state_index] = fixed_reward(
                        state, action, next_state)

        # put all features that have a value other than 0 associated with it into a list for a consistent indexing purpose
        features = set()
        for state in states.keys():
            for action in actions.keys():
                for next_state in mdp.next_state_dist(state, action).support:
                    features.update(featurizer(
                        state, action, next_state).keys())
        # OR all features must be listed in the dictionary no matter whether they are 0
        features = sorted(features)
        num_features = len(features)
        # update feature_matrix using user-defined featurizer
        feature_matrix = np.zeros(
            (num_states, num_actions, num_states, num_features))
        for state, state_index in states.items():
            for action, action_index in actions.items():
                for next_state in mdp.next_state_dist(state, action).support:
                    next_state_index = states.get(next_state)
                    for feat, feat_val in featurizer(state, action, next_state).items():
                        assert isinstance(
                            feat_val, (int, float)), "value associated with feature needs to be a number"
                        feat_index = features.index(feat)
                        feature_matrix[state_index, action_index,
                                       next_state_index, feat_index] = feat_val

        self.feature_matrix = torch.tensor(feature_matrix)
        self.fixed_reward_matrix = torch.tensor(fixed_reward_matrix)

    def learn(self, trajectories: TrajectoryDataset):
        '''
        Parameters:
          trajectories: TrajectoryDataset trajectories
        Output:
          r_weights: the learned reward weights
          policy: policy associated with the final r_weights
          all_losses: array of loss per epoch
        '''
        all_losses = []
        trajectory_weight = 1/len(trajectories)
        trajs_dataloader = DataLoader(
            trajectories, batch_size=self.batch_size, shuffle=True, collate_fn=self.__collate_fn)

        # randomize weights
        r_weights = torch.tensor(
            np.random.randn(self.feature_matrix.shape[-1]))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device}")
        print(f"Inital reward weights: {r_weights}\n")

        # try using the true reward weights to debug - Checked. Has basically the same rewards
        # r_weights = torch.tensor([float(mdp_params['feature_rewards'][f]) for f in features]).double()

        r_weights.requires_grad = True
        discount_rate = torch.tensor(self.mdp.discount_rate)
        transition_matrix = torch.tensor(self.mdp.transition_matrix)

        optimizer = torch.optim.SGD(
            [r_weights], lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

        size = len(trajs_dataloader.dataset)

        for _ in range(self.epochs):
            for batch, sample in enumerate(trajs_dataloader):
                optimizer.zero_grad()

                # build the reward matrix of the entire MDP using current reward weights

                # We do not have the reward weights from MDP bc the reward weights are what we are optimizing. Thus, use r_weights
                feature_reward_matrix = torch.einsum(
                    "sanf,f->san",
                    self.feature_matrix,
                    r_weights
                )

                # construct the final reward function by taking into account each action's step cost
                reward_matrix = feature_reward_matrix + self.fixed_reward_matrix

                # anything leading to the terminal state ({-1, -1}) has zero reward
                terminal_index = self.mdp.state_index.get(
                    frozendict({'x': -1, 'y': -1}))
                reward_matrix[:, :, terminal_index] = 0

                # compute the optimal policy for the current reward function
                policy = entropy_regularized_policy_iteration(
                    transition_matrix, reward_matrix, discount_rate, self.entropy_weight, self.planning_iters).policy
                loss = torch.tensor(0.0)

                for state_traj, action_traj in sample:
                    for index in range(len(state_traj)):
                        state = state_traj[index]
                        action = action_traj[index]
                        state_index = self.mdp.state_index.get(state)
                        action_index = self.mdp.action_index.get(action)
                        loss -= torch.log(policy[state_index,
                                                 action_index]) * trajectory_weight
                loss.backward()
                optimizer.step()

                if batch % 10000 == 0:
                    loss, current = loss.item(), batch * len(sample)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            all_losses.append(loss.item())

        print(f"Final reward weights: {r_weights}\n")

        # compute policy from learned weights
        feature_reward_matrix = torch.einsum(
            "sanf,f->san",
            self.feature_matrix,
            r_weights
        )

        reward_matrix = feature_reward_matrix + self.fixed_reward_matrix
        terminal_index = self.mdp.state_index.get(
            frozendict({'x': -1, 'y': -1}))
        reward_matrix[:, :, terminal_index] = 0
        policy = entropy_regularized_policy_iteration(
            transition_matrix, reward_matrix, discount_rate, self.entropy_weight, self.planning_iters)

        return r_weights, policy, all_losses

    def get_feature_matrix(self):
        return self.feature_matrix

    def get_fixed_reward_matrix(self):
        return self.fixed_reward_matrix

    def __collate_fn(self, batch):
        return batch


class ImitationLearning(Algorithm):
    def __init__(self, mdp, state_featurizer, action_featurizer, model, loss_fn, optimizer, scheduler=None, batch_size=128, epochs=50):
        '''
        Parameters:
          mdp: a representation of the Markov Decision Process
          state_featurizer: func(state) -> dictionary w/ features as keys and float|int associated with each feature as values; unspecified features will have a value of 0
          action_featurizer: func(action) -> integer
          model: PyTorch neural network
          loss_fn: PyTorch loss function
          optimizer: PyTorch optimizer
          scheduler: PyTorch scheduler (varies learning rate at specific epochs)
          batch_size: size of a batch of data for training
          epochs: number of epochs for learning the reward function
        Internal Output:
          state_feature_matrix: S x F where the arr[i, :] = the features of state i, F = number of features
        '''

        self.mdp = mdp
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.epochs = epochs

        # num_states = len(self.mdp.state_list)
        # states = mdp.state_index

        # # put all features that have a value other than 0 associated with it into a list for a consistent indexing purpose
        # features = set()
        # for state in states.keys():
        #     features.update(state_featurizer(state).keys())

        # features = sorted(features)
        # num_features = len(features)
        # # update feature_matrix using user-defined state_featurizer
        # state_feature_matrix = np.zeros(num_states, num_features)
        # for state, state_index in states.items():
        #     for feat, feat_val in state_featurizer(state).items():
        #         assert isinstance(
        #             feat_val, (int, float)), "value associated with feature needs to be a number"
        #         feat_index = features.index(feat)
        #         state_feature_matrix[state_index, feat_index] = feat_val

        # self.state_feature_matrix = state_feature_matrix
        # self.action_featurizer = action_featurizer

    def learn(self, trajectories: FeaturesDataset):
        '''
        Parameters:
          trajectories: array of trajectories which are dictionaries containing the state and action trajectory
        Output:
          model: trained PyTorch model
        '''
        all_losses = []
        trajs_dataloader = DataLoader(
            trajectories, batch_size=self.batch_size, shuffle=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device}\n")

        self.model = self.model.to(device)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.__train(trajs_dataloader, device, all_losses)
            print()
            if self.scheduler is not None:
                self.scheduler.step()
        print("Done!")

        return self.model, all_losses

    def __train(self, dataloader, device, all_losses):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 10000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        all_losses.append(loss.item())
