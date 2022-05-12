hparams = {
    'file_name': "runs_uf4_cnn_lstm.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn_lstm'],
            'env': ['uf4m'],
            'slip_prob': [0.1],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [32],
            'discounting': [0.99],
            'step_size': [0.0000001, 0.000001, 0.00001, 0.0001],
            'trunc': [1, 5, 10],
            'action_cond': ['cat'],
            'uncertainty_decay': [1., 0.95, 0.9, 0.85],
            'buffer_size': [100000],
            'total_steps': [2000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5)]
            # 'seed': [(i + 2020) for i in range(10)]
        }]
}