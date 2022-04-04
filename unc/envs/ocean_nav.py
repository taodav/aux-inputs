import numpy as np


from unc.envs.base import Environment


class OceanNav(Environment):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)
    group_mapping = [[(7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5)]]

    def __init__(self, rng: np.random.RandomState, size: int = 9,
                 obs_size: int = 5, half_efficiency_distance: float = 3):
        super(OceanNav, self).__init__()
        self.size = size
        self.rng = rng
        self.obs_size = obs_size
        self.half_efficiency_distance = half_efficiency_distance
        self.position = None
        self.obstacle_map = np.array((self.size, self.size))
        self.wind_map = np.array((self.size, self.size))


