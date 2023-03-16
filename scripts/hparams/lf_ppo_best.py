hparams = {
    'file_name': "lobster_ppo_best.txt",
    'args':
        [{
            'algo': ['ppo'],
            'arch': ['actor_critic'],
            'env': ['2', '2o', '2d', '2pb'],
            'discounting': [0.9],
            'n_hidden': [10],
            'step_size': [0.01],
            'total_steps': [250000],
            'max_episode_steps': [200],
            'layers': [1],
            'n_particles': [100],
            'seed': [(i + 2020) for i in range(40, 100)]
        }]
}
