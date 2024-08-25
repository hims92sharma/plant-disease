import tensorflow as tf
from keras import layers, models
from keras.losses import binary_crossentropy
from keras.datasets import mnist
from keras.backend import backend as K
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define VAE architecture
latent_dim = 2


def build_vae(encoder, decoder):
    x = layers.Input(shape=(image_size, image_size, 1))
    z_mean, z_log_var, z = encoder(x)
    print("z_mean shape:", z_mean.shape)
    print("z_log_var shape:", z_log_var.shape)
    x_hat = decoder(z)

    model = models.Model(x, x_hat, name='vae')
    return model


def build_encoder():
    x = layers.Input(shape=(image_size, image_size, 1))
    h = layers.Conv2D(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    h = layers.Conv2D(64, kernel_size=3, activation='relu', strides=2, padding='same')(h)
    h = layers.Flatten()(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    model = models.Model(x, [z_mean, z_log_var, z], name='encoder')
    return model


def build_decoder():
    z = layers.Input(shape=(latent_dim,))
    h = layers.Dense(7 * 7 * 64, activation='relu')(z)
    h = layers.Reshape((7, 7, 64))(h)
    h = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', strides=2, padding='same')(h)
    h = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(h)
    x_hat = layers.Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(h)

    model = models.Model(z, x_hat, name='decoder')
    return model


# Define VAE components
encoder = build_encoder()
decoder = build_decoder()
vae_input = layers.Input(shape=(image_size, image_size, 1))
z = encoder(vae_input)
vae_output = decoder(z)
vae = models.Model(vae_input, vae_output, name='vae')


# Define VAE loss function
def vae_loss(x, x_hat, z_mean, z_log_var):
    x = K.flatten(x)
    x_hat = K.flatten(x_hat)
    xent_loss = image_size * image_size * binary_crossentropy(x, x_hat)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)


# Compile VAE model
vae.compile(optimizer='adam', loss=lambda x, x_hat: vae_loss(x, x_hat, z_mean, z_log_var))


# Train the VAE
vae.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Generate images using the trained VAE
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = np.linspace(-1.5, 1.5, n)
grid_y = np.linspace(-1.5, 1.5, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

# Plot the generated digits
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
