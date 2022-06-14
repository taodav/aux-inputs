hparams = {
    'file_name': "lobster_nn_pf_sweep.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['nn', 'linear'],
            'env': ['2pb'],
            'discounting': [0.9],
            'n_hidden': [5],
            'platform': ['cpu'],
            'n_particles': [100],
            'step_size': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            # 'step_size': [0.01, 0.1],
            'total_steps': [250000],
            'max_episode_steps': [200],
            'seed': [(i + 2020) for i in range(30)]
        }]
}
