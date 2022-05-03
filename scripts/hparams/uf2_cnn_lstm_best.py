hparams = {
    'file_name': "runs_uf2_cnn_lstm_best.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn_lstm'],
            'env': ['uf2p'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [32],
            'step_size': [0.0001],
            'trunc': [10],
            'action_cond': ['cat'],
            'uncertainty_decay': [1.],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5, 30)]
        },
        {
            'algo': ['sarsa'],
            'arch': ['cnn_lstm'],
            'env': ['uf2m'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [32],
            'step_size': [1e-5],
            'trunc': [10],
            'action_cond': ['cat'],
            'uncertainty_decay': [1.],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5, 30)]
        }
        ]
}
