hparams = {
    'file_name': "runs_uf3_cnn_lstm.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn_lstm'],
            'env': ['uf3m'],
            'slip_prob': [0., 0.1],
            # 'distance_noise': [False, True],
            # 'distance_unc_encoding': [False, True],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [32],
            'step_size': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001],
            # 'step_size': [0.0001],
            'trunc': [10],
            'action_cond': ['cat'],
            'uncertainty_decay': [1.],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5)]
            # 'seed': [(i + 2020) for i in range(10)]
        }]
}
