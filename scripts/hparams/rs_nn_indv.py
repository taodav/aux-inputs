hparams = {
    'file_name': "runs_rs_lstm_no_cat.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'env': ['rg'],
            'arch': ['lstm'],
            'batch_size': [64],
            'discounting': [0.99],
            'p_prefilled': [0.0],
            'trunc': [10],
            'action_cond': [None],
            'replay': [True],
            'step_size': [0.001, 0.0001, 0.00001],
            'half_efficiency_distance': [5.],
            'buffer_size': [10000, 100000],
            'offline_eval_freq': [2500],
            'total_steps': [1500000],
            'rock_obs_init': [0.0],
            'seed': [(i + 2020) for i in range(10)]
        }]
}
