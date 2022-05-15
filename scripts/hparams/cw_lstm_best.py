hparams = {
    'file_name': "runs_compass_lstm_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'arch': ['lstm'],
                'env': ['f'],
                'replay': [True],
                'size': [9],
                'action_cond': ['cat'],
                'trunc': [20],
                'step_size': [0.0001],
                'offline_eval_freq': [2000],
                'batch_size': [64],
                'buffer_size': [100000],
                'total_steps': [1000000],
                'seed': [(i + 2020) for i in range(30)]
            }
        ]
}
