hparams = {
    'file_name': "runs_rs_best.txt",
    'args':
        [
            {
                'algo': ['sarsa'],
                'env': ['rg'],
                'arch': ['nn'],
                'batch_size': [64],
                'discounting': [0.99],
                'p_prefilled': [0.0],
                'replay': [True],
                'step_size': [0.001],
                'half_efficiency_distance': [5.],
                'buffer_size': [100000],
                'offline_eval_freq': [2500],
                'total_steps': [1500000],
                'rock_obs_init': [0.5],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 40)]
            },
            {
                'algo': ['sarsa'],
                'env': ['rg'],
                'arch': ['lstm'],
                'batch_size': [64],
                'discounting': [0.99],
                'p_prefilled': [0.0],
                'trunc': [10],
                'action_cond': ['cat'],
                'replay': [True],
                'step_size': [0.0001],
                'half_efficiency_distance': [5.],
                'buffer_size': [100000],
                'offline_eval_freq': [2500],
                'total_steps': [1500000],
                'rock_obs_init': [0.0],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 40)]
            },
            {
                'algo': ['sarsa'],
                'env': ['rxg'],
                'arch': ['nn'],
                'batch_size': [64],
                'discounting': [0.99],
                'p_prefilled': [0.0],
                'replay': [True],
                'step_size': [0.001],
                'half_efficiency_distance': [5.],
                'buffer_size': [100000],
                'offline_eval_freq': [2500],
                'total_steps': [1500000],
                'rock_obs_init': [0.5],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 40)]
            },
            {
                'algo': ['sarsa'],
                'env': ['rpg'],
                'arch': ['nn'],
                'n_particles': [100],
                'batch_size': [64],
                'discounting': [0.99],
                'p_prefilled': [0.0],
                'replay': [True],
                'step_size': [0.001],
                'half_efficiency_distance': [5.],
                'buffer_size': [100000],
                'offline_eval_freq': [2500],
                'total_steps': [1500000],
                'rock_obs_init': [0.0],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 40)]
            },
            {
                'algo': ['sarsa'],
                'env': ['rsg'],
                'arch': ['nn'],
                'batch_size': [64],
                'discounting': [0.99],
                'p_prefilled': [0.0],
                'replay': [True],
                'step_size': [0.0001],
                'half_efficiency_distance': [5.],
                'buffer_size': [100000],
                'offline_eval_freq': [2500],
                'total_steps': [1500000],
                'rock_obs_init': [0.0],
                # 'seed': [(i + 2020) for i in range(30)]
                'seed': [(i + 2020) for i in range(30, 40)]
            },

        ]
}
