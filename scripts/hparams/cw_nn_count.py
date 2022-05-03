hparams = {
    'file_name': "runs_compass_state_count.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['nn'],
            'env': ['fc'],
            'batch_size': [64],
            'replay': [False],
            'size': [9],
            'trunc': [0],
            'count_decay': [0.75, 0.9, 1.],
            'step_size': [0.001, 0.0001, 0.00001],
            'buffer_size': [10000],
            'total_steps': [1000000],
            'seed': [(i + 2020) for i in range(10)]
        }]
}
