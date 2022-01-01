from torch.utils.data import Dataset
import numpy as np


# use transforms for transforming data for supervised learning
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, transform=None, target_transform=None):
        '''
        Parameters:
          trajectories: array of trajectories which are dictionaries containing 'state_traj' and 'action_traj' keys with tuple values
          trajectories format example = [{'state_traj': (1, 2, 3, 4, 5), 'action_traj': (2, 7, 9, 5, 1)}]
        '''
        assert self.__check(
            trajectories), "trajectories elements must conform to indicated structure"

        trajectories = np.array(trajectories)

        self.state_trajectories = []
        self.action_trajectories = []

        for trajectory in trajectories:
            self.state_trajectories.append(trajectory.get('state_traj'))
            self.action_trajectories.append(trajectory.get('action_traj'))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.state_trajectories)

    # an item is 1 trajectory (state_traj and corresponding action_traj)
    def __getitem__(self, idx):
        state_traj = self.state_trajectories[idx]
        action_traj = self.action_trajectories[idx]
        if self.transform:
            state_traj = self.transform(state_traj)
        if self.target_transform:
            action_traj = self.target_transform(action_traj)
        return state_traj, action_traj

    def __check(self, trajectories):
        str_arr = ['state_traj', 'action_traj']
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                return False
            if len(trajectory.keys()) != 2:
                return False
            for traj_name, traj_sequence in trajectory.items():
                if traj_name not in str_arr:
                    return False
                if not isinstance(traj_sequence, tuple):
                    return False
        return True


class FeaturesDataset(Dataset):
    # trajectories format =  [{'state_traj': (1, 2, 3, 4, 5), 'action_traj': (2, 7, 9, 5, 1)}]
    def __init__(self, mdp, trajectories, transform=None, target_transform=None):
        self.mdp = mdp
        self.color_features = sorted(mdp.feature_locations.keys())
        self.color_features = [f for f in self.color_features if f not in 'gs']
        self.state_features, self.actions = self.__getFeaturesActions(
            trajectories=trajectories)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        state_feature = self.state_features[idx]
        action = self.actions[idx]
        if self.transform:
            state_feature = self.transform(state_feature)
        if self.target_transform:
            action = self.target_transform(action)
        return state_feature, action

    # returns a label 0, 1, 2, 3, or 4
    def __convertAction(self, action):
        dy = action.get('dy')
        dx = action.get('dx')
        if dy == 1:
            return 0
        elif dy == -1:
            return 1
        elif dx == 1:
            return 2
        elif dx == -1:
            return 3
        else:
            return 4

    # state feature vector: gets the features of a state which are just (x, y) coordinates
    # parameter state in form (frozendict({'x': 9, 'y': 0})
    # returns a 2D array [[x, y, state_identifier]]
    def __getStateFeature(self, state):
        f = self.mdp.location_features.get(state, '.')
        if f in self.color_features:
            color_index = self.color_features.index(f)
        else:
            color_index = len(self.color_features)
        return [[state.get('x'), state.get('y'), color_index]]

    # get state features (x-coord, y-coord, state_identifier) and actions for each state in each trajectory
    # return numpy arrays, all_features and all_actions.
    # all_features: a 2D array where each row is a state (a sample) and column 0 is x-coordinate, column 1 is y-coordinate, column
    # 2 is state identifier
    # all_actions: a 1D array where each row corresponds to the action to take in the same row of all_features
    def __getFeaturesActions(self, trajectories):
        all_features = []  # converting all states to features for training data
        all_actions = []  # storing the actions taken in each state in the same order
        for trajectory in trajectories:
            # assuming trajectory is an array of tuples
            state_traj = trajectory.get("state_traj")  # grab state trajectory
            action_traj = trajectory.get(
                "action_traj")  # grab action trajectory
            for state in state_traj:
                # states are format (frozendict({'x': 9, 'y': 0})
                # convert states to (x, y, state_identifier)
                state_feature = self.__getStateFeature(state)
                all_features.append(state_feature)
            for action in action_traj:
                # actions are format frozendict({'dy': 1, 'dx': 0})
                # convert action to single dimension (an integer)
                conv_action = self.__convertAction(action)
                all_actions.append(conv_action)

        # all_features must be a 2d array of floats and all_actions must be an array of integers
        return np.array(all_features, dtype='f'), np.array(all_actions)
