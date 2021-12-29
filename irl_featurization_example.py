class MyIRLAlgorithm(InverseRL):
    def __init__(
        self,
        mdp,
        featurizer,
        fixed_reward
    ):
    """
    How to do featureizer?
    featurizer(state, action, next_state) -> {'next_is_blue': 1, 'next_is_green': 0} GOOD
    featurizer(state, action, next_state) -> {'next_color': 5.1, 'move_left': 1} GOOD
    featurizer(state, action, next_state) -> {'next_color': 'blue'} BAD! Values must be numbers

    Reward featurizer
    fixed_reward(state, action, next_state) -> float
    """
        n_states = len(mdp.state_list)
        n_actions = len(mdp.action_list)

        # building the fixed reward matrix
        fixed_reward_matrix = np.zeros(n_states, n_actions, n_states)
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                for ns in mdp.next_state_dist(s, a).support:
                    nsi = mdp.state_list.index(ns)
                    fixed_reward_matrix[si, ai, nsi] = fixed_reward(s, a, ns)

        # building the feature matrix - first, get all features, then fill in matrix
        all_features = set([])
        for s in mdp.state_list:
            for a in mdp.action_list:
                for ns in mdp.next_state_dist(s, a).support:
                    sans_features = featurizer(s, a, ns)
                    all_features.update(sans_features.keys())
        feature_names = sorted(all_features) #ensure that order is stable
        n_features = len(feature_names)

        feature_matrix = np.zeros(n_states, n_actions, n_states, n_features)
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                for ns in mdp.next_state_dist(s, a).support:
                    sans_features = featurizer(s, a, ns)
                    for feature, value in sans_features.items():
                        assert isinstance(value, (int, float))
                        fi = feature_names.index(feature)
                        feature_matrix[si, ai, nsi, fi] = value
