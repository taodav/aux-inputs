hparams = {
    'file_name': "runs_uf8_cnn.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf8m', 'uf8a'],
            'slip_prob': [0.1],
            # 'random_reward_start': [True],
            'max_episode_steps': [1000],
            'batch_size': [64],
            # 'replay': [True, False],
            'replay': [True],
            'n_hidden': [64],
            'discounting': [0.99],
            'step_size': [0.0000001, 0.000001, 0.00001, 0.0001],
            'uncertainty_decay': [1., 0.95, 0.85, 0.65],
            'buffer_size': [100000],
            # 'total_steps': [6000000],
            'total_steps': [12000000],
            'offline_eval_freq': [10000],
            'checkpoint_freq': [20000],
            'save_all_checkpoints': [False],
            # 'total_steps': [3000000],
            # 'offline_eval_freq': [5000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5)]
            # 'seed': [(i + 2020) for i in range(10)]
        }]
}
