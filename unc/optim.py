import optax


def get_optimizer(optimizer: str, step_size: float):
    if optimizer == "adam":
        return optax.adam(step_size)
    elif optimizer == "sgd":
        return optax.sgd(step_size)
    else:
        raise NotImplementedError

