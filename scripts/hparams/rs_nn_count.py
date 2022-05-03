hparams = {
    'file_name': "runs_rs_state_count.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['nn'],
            'env': ['rc'],
            'n_particles': [100],
            'batch_size': [64],
            'discounting': [0.99],
            'p_prefilled': [0.0],
            'replay': [True],
            'step_size': [0.001, 0.0001, 0.00001],
            'buffer_size': [10000, 100000],
            'total_steps': [1500000],
            'unnormalized_counts': [True],
            'count_decay': [0.75, 0.9, 1.],
            'seed': [(i + 2020) for i in range(10)]
        }]
}
