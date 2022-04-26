hparams = {
    'file_name': "runs_compass_lstm_hs_pf.txt",
    'args':
        {
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf1m', 'uf1a'],
            'max_episode_steps': [500],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [100],
            'step_size': [0.00001, 0.0001, 0.001, 0.01],
            'uncertainty_decay': [1., 0.9],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [1000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(10)]
        }
}
