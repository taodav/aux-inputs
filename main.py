import torch
import numpy as np

from unc.envs import get_env
from unc.args import Args, get_results_fname
from unc.trainer import Trainer
from unc.models import QNetwork
from unc.agents import SarsaAgent
from unc.utils import save_info


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

    # Some argument post-processing
    results_fname = get_results_fname(args)
    args.results_fname = results_fname

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # Initializing our environment
    # TODO: TEST THIS
    train_env = get_env(args.seed, env_str=args.env, blur_prob=args.blur_prob, random_start=args.random_start)

    # Initialize model, optimizer and agent
    model = QNetwork(train_env.observation_space.shape[0], args.n_hidden, train_env.action_space.n).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.step_size)
    agent = SarsaAgent(model, optimizer, train_env.action_space.n, rng,
                       args)

    # Initialize our trainer
    trainer = Trainer(args, agent, train_env)
    trainer.reset()

    # Train!
    trainer.train()

    # Save results
    results_fname = get_results_fname(args)
    results_path = args.results_dir / args.results_fname
    print(f"Saving results to {results_path}")
    save_info(results_path, trainer.get_info())


