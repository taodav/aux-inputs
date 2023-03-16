hparams = {
    'file_name': "lobster_linear_best.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['linear'],
            'env': ['2', '2o', '2d', '2pb'],
            'discounting': [0.9],
            'epsilon': [0.1],
            'n_hidden': [5],
            'step_size': [0.001],
            'total_steps': [250000],
            'max_episode_steps': [200],
            'n_particles': [100],
            'seed': [(i + 2020) for i in range(30, 60)]
        }]
}
