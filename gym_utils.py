def get_box_bound(gym_box):
    high = gym_box.high
    low  = gym_box.low
    return low, high


def get_box_dim(gym_box):
    return gym_box.shape[0]


def get_state_dimension(gym_env):
    return get_box_dim(gym_env.observation_space)


def get_action_dimension(gym_env):
    return get_box_dim(gym_env.action_space)


def get_state_bound(gym_env):
    return get_box_bound(gym_env.observation_space)


def get_action_bound(gym_env):
    return get_box_bound(gym_env.action_space)