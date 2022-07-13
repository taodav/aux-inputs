from jax import random
import sys
from pathlib import Path
from pickle import UnpicklingError

from .trainer import Trainer
from .buffer_trainer import BufferTrainer

from unc.agents import Agent
from unc.args import Args
from unc.envs import Environment
from unc.utils.gvfs import GeneralValueFunction
from unc.utils.replay import ReplayBuffer
from unc.utils.files import load_checkpoint


def get_or_load_trainer(args: Args, rand_key: random.PRNGKey, agent: Agent,
                        train_env: Environment, test_env: Environment,
                        checkpoint_dir: Path = None, prefilled_buffer: ReplayBuffer = None,
                        gvf: GeneralValueFunction = None):

    if checkpoint_dir is not None and checkpoint_dir.is_dir():
        # this means the checkpoint dir exists. We need to get the latest ckpt.
        sorted_ckpts = sorted(list(checkpoint_dir.iterdir()), key=lambda x: int(x.stem))
        if sorted_ckpts:
            # if we're here, the stem of this path should be the number of steps done.
            checkpoint_path = sorted_ckpts[-1]
            latest_checkpoint_steps = int(checkpoint_path.stem)

            if latest_checkpoint_steps <= args.total_steps:
                # continue training
                trainer = None
                while len(sorted_ckpts) > 0 and trainer is None:
                    try:
                        checkpoint_path = sorted_ckpts[-1]
                        print(f"Loading trainer {checkpoint_path}")
                        trainer = load_checkpoint(checkpoint_path)
                    except UnpicklingError as e:
                        print(f"Pickling error for file {checkpoint_path}, deleting and trying next file", file=sys.stderr)
                        checkpoint_path.unlink(missing_ok=True)
                        sorted_ckpts = sorted_ckpts[:-1]


                if trainer is not None:
                    # we have to replace all of our
                    trainer.total_steps = args.total_steps
                    trainer.offline_eval_freq = args.offline_eval_freq
                    trainer.test_eps = args.test_eps
                    trainer.test_episodes = args.test_episodes
                    trainer.checkpoint_freq = args.checkpoint_freq
                    trainer.checkpoint_dir = checkpoint_dir
                    trainer.save_all_checkpoints = args.save_all_checkpoints
                    return trainer, rand_key


    if args.replay:
        rand_key, buffer_rand_key = random.split(rand_key, 2)
        trainer = BufferTrainer(args, agent, train_env, test_env, buffer_rand_key,
                                checkpoint_dir=checkpoint_dir, prefilled_buffer=prefilled_buffer, gvf=gvf)
    else:
        trainer = Trainer(args, agent, train_env, test_env, checkpoint_dir=checkpoint_dir, gvf=gvf)

    trainer.reset()

    return trainer, rand_key

