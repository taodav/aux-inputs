hparams = {
    'file_name': "runs_uf2m_cnn_best.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf2a'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [64],
            'step_size': [0.0001],
            'uncertainty_decay': [1.],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5, 30)]
        },
        {
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf2m'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [64],
            'step_size': [1e-5],
            'uncertainty_decay': [1.],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5, 30)]
        },
        {
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf2m'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [64],
            'step_size': [0.001],
            'uncertainty_decay': [0.9],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5, 30)]
        }
        ]
}
