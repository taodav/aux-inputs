hparams = {
    'file_name': "runs_uf4_cnn_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf4m'],
                'slip_prob': [0.1],
                # 'random_reward_start': [True],
                'max_episode_steps': [1000],
                'batch_size': [64],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-6],
                'uncertainty_decay': [0.95],
                'buffer_size': [100000],
                'total_steps': [2000000],
                'offline_eval_freq': [2000],
                'platform': ["gpu"],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 35)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf4m'],
                'slip_prob': [0.1],
                # 'random_reward_start': [True],
                'max_episode_steps': [1000],
                'batch_size': [64],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-5],
                'uncertainty_decay': [1.],
                'buffer_size': [100000],
                'total_steps': [2000000],
                'offline_eval_freq': [2000],
                'platform': ["gpu"],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 35)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf4a'],
                'slip_prob': [0.1],
                'max_episode_steps': [1000],
                'batch_size': [64],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-5],
                'uncertainty_decay': [1.],
                'buffer_size': [100000],
                'total_steps': [2000000],
                'offline_eval_freq': [2000],
                'platform': ["gpu"],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 35)]
            },
        ]
}
