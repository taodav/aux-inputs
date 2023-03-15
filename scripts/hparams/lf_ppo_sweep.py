hparams = {
    'file_name': "lobster_ppo_sweep.txt",
    'args':
        [{
            'algo': ['ppo'],
            'arch': ['actor_critic'],
            'env': ['2', '2o', '2d', '2pb'],
            'discounting': [0.9],
            'n_hidden': [5, 10],
            # 'step_size': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'step_size': [0.1],
            'total_steps': [500000],
            'max_episode_steps': [200],
            'layers': [1],
            'n_particles': [100],
            'seed': [(i + 2020) for i in range(10)]
        }]
}
