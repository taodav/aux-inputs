hparams = {
    'file_name': "runs_uf8_cnn_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf8a'],
                'slip_prob': [0.1],
                'max_episode_steps': [1000],
                'batch_size': [64],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-5],
                'uncertainty_decay': [1.],
                'buffer_size': [100000],
                'total_steps': [6000000],
                'offline_eval_freq': [10000],
                'checkpoint_freq': [20000],
                'save_all_checkpoints': [False],
                'platform': ["gpu"],
                'seed': [(i + 2020) for i in range(5, 30)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf8m'],
                'slip_prob': [0.1],
                'max_episode_steps': [1000],
                'batch_size': [64],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-6],
                'uncertainty_decay': [1., 0.85],
                'buffer_size': [100000],
                'total_steps': [6000000],
                'offline_eval_freq': [10000],
                'checkpoint_freq': [20000],
                'save_all_checkpoints': [False],
                'platform': ["gpu"],
                'seed': [(i + 2020) for i in range(5, 30)]
            }
        ]
}