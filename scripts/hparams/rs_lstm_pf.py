hparams = {
    'file_name': "runs_rs_lstm_pf.txt",
    # 'file_name': "runs_rs_lstm_hed_5.txt",
    'args':
        {
            'algo': ['sarsa'],
            'arch': ['lstm'],
            'env': ['rpg'],
            'n_particles': [100],
            'batch_size': [64],
            'discounting': [0.99],
            'p_prefilled': [0.0],
            'replay': [True],
            'step_size': [0.001, 0.0001, 0.00001],
            'trunc': [10, 20],
            'buffer_size': [10000, 100000],
            'total_steps': [1500000],
            # 'half_efficiency_distance': [5.],
            # 'action_cond': [None, 'cat'],
            'seed': [(i + 2020) for i in range(10)]
        }
}
