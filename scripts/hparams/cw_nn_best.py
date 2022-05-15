hparams = {
    'file_name': "runs_compass_nn_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'arch': ['nn'],
                'env': ['f'],
                'replay': [False],
                'size': [9],
                'step_size': [0.0001],
                'offline_eval_freq': [2000],
                'total_steps': [1000000],
                'seed': [(i + 2020) for i in range(30)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['nn'],
                'env': ['fpg'],
                'replay': [False],
                'size': [9],
                'step_size': [0.001],
                'offline_eval_freq': [2000],
                'total_steps': [1000000],
                'seed': [(i + 2020) for i in range(30)]
            },
            {
                'algo': ['sarsa'],
                'arch': ['nn'],
                'env': ['fsg'],
                'replay': [False],
                'size': [9],
                'step_size': [0.0001],
                'offline_eval_freq': [2000],
                'total_steps': [1000000],
                'seed': [(i + 2020) for i in range(30)]
            },
        ]
}
