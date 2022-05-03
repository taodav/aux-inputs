hparams = {
    'file_name': "runs_uf2m_cnn.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['cnn'],
            'env': ['uf2m', 'uf2a'],
            'max_episode_steps': [1000],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [64],
            # 'step_size': [0.00001, 0.0001, 0.001, 0.01],
            'step_size': [0.000001, 0.0000001],
            'uncertainty_decay': [1., 0.9],
            'buffer_size': [100000],
            'total_steps': [1000000],
            'offline_eval_freq': [2000],
            'platform': ["gpu"],
            'seed': [(i + 2020) for i in range(5)]
            # 'seed': [(i + 2020) for i in range(10)]
        }]
}
