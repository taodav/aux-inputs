hparams = {
    'file_name': "lobster_nn_gvf_fixed_sweep.txt",
    'entry': 'scripts/train_lobster_gvfs.py',
    'args':
        [{
            'algo': ['sarsa'],
            'arch': ['nn'],
            'env': ['2t'],
            'discounting': [0.9],
            'n_hidden': [40],
            'platform': ['cpu'],
            'epsilon': [1.],
            'step_size': [10**(-i) for i in range(3, 8)],
            'total_steps': [500000],
            'max_episode_steps': [200],
            'seed': [(i + 2020) for i in range(30)]
        }]
}
