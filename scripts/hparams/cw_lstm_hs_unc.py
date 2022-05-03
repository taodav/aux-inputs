hparams = {
    'file_name': "runs_compass_lstm_hs_pf.txt",
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['lstm'],
            'env': ['f'],
            'batch_size': [64],
            'replay': [True],
            'n_hidden': [20],
            'k_rnn_hs': [10],
            'same_k_rnn_params': [True],
            'trunc': [10],
            'size': [9],
            'init_hidden_var': [0.1, 0.5],
            'step_size': [0.00001, 0.0001, 0.001],
            'value_step_size': [0.0001, 0.00025, 0.0005],
            'action_cond': ['cat', None],
            'buffer_size': [10000, 100000],
            'total_steps': [1000000],
            'seed': [(i + 2020) for i in range(10)]
        }]
}
