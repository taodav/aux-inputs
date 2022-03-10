hparams = {
    'file_name': "runs_compass_lstm_pf.txt",
    'args':
    {
        'algo': ['sarsa'],
        'arch': ['lstm'],
        'env': ['fpg'],
        'batch_size': [64],
        'replay': [True],
        'trunc': [10, 20],
        'size': [9],
        'step_size': [0.001, 0.0001, 0.00001],
        'buffer_size': [10000, 100000],
        'total_steps': [1000000],
        'action_cond': ['cat', None],
        'seed': [(i + 2020) for i in range(10)]
     }
}