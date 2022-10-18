hparams = {
    'file_name': "lobster_linear_dummy_gvf_sweep.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['linear'],
            'env': ['2d'],
            'discounting': [0.9],
            'platform': ['cpu'],
            'epsilon': [0.1],
            'step_size': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'total_steps': [250000],
            'max_episode_steps': [200],
            'seed': [(i + 2020) for i in range(30)]
        }]
}
