hparams = {
    'file_name': "runs_uf8_cnn_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf8m'],
                'slip_prob': [0.1],
                # 'random_reward_start': [True],
                'max_episode_steps': [1000],
                'batch_size': [64],
                # 'replay': [True, False],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-6],
                'uncertainty_decay': [1.],
                'buffer_size': [100000],
                'total_steps': [3000000],
                'offline_eval_freq': [5000],
                'platform': ["gpu"],
                'seed': [(i + 2020) for i in range(30)]
                # 'seed': [(i + 2020) for i in range(10)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['cnn'],
                'env': ['uf8m'],
                'slip_prob': [0.1],
                # 'random_reward_start': [True],
                'max_episode_steps': [1000],
                'batch_size': [64],
                # 'replay': [True, False],
                'replay': [True],
                'n_hidden': [64],
                'discounting': [0.99],
                'step_size': [1e-6],
                'uncertainty_decay': [0.85],
                'buffer_size': [100000],
                'total_steps': [3000000],
                'offline_eval_freq': [5000],
                'platform': ["gpu"],
                'seed': [(i + 2020) for i in range(30)]
                # 'seed': [(i + 2020) for i in range(10)]
            }
        ]
}
