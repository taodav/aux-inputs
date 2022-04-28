import jax
import jax.numpy as jnp
import haiku as hk


class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), padding="SAME")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(5)

    def __call__(self, x_batch):
        x = self.conv1(x_batch)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = jax.nn.softmax(x)
        return x

def ConvNet(x):
    cnn = CNN()
    return cnn(x)

if __name__ == "__main__":
    jax.config.update('jax_platform_name', "gpu")

    dummy = jnp.zeros((1,28,28,1))
    conv_net = hk.transform(ConvNet)
    rng = jax.random.PRNGKey(42)

    params = conv_net.init(rng, dummy)

    print("Initialized successfully.")
