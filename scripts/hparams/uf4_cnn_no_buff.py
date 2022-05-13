hparams = {
    'file_name': "runs_uf4_cnn_no_buff.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn', 'cnn_lstm'],
            'env': ['uf4m'],
            'slip_prob': [0.1],
            'max_episode_steps': [1000],
            'n_hidden': [64],
            'discounting': [0.99],
            'step_size': [0.000001],
            'uncertainty_decay': [1., 0.9],
            'trunc': [1, 5, 10],
            'buffer_size': [100000],
            'total_steps': [2000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(3)]
            # 'seed': [(i + 2020) for i in range(10)]
        }]
}
