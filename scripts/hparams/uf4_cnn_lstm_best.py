hparams = {
    'file_name': "runs_uf4_cnn_lstm_best.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn_lstm'],
            'env': ['uf4m'],
            'slip_prob': [0.1],
            # 'random_reward_start': [True],
            'max_episode_steps': [1000],
            'batch_size': [64],
            # 'replay': [True, False],
            'replay': [True],
            'n_hidden': [32],
            'discounting': [0.99],
            'step_size': [1e-5],
            'trunc': [1],
            'action_cond': ['cat'],
            'uncertainty_decay': [1., 0.95],
            'buffer_size': [100000],
            'total_steps': [2000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            # 'seed': [(i + 2020) for i in range(30)]
            'seed': [(i + 2020) for i in range(30, 35)]
        }]
}
