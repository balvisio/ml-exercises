from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mixed_precision = True
if mixed_precision:
    keras.mixed_precision.set_global_policy("mixed_float16")

"""
To download the dataset:
mkdir celeba_gan
gdown --id 1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684 -O celeba_gan/data.zip
unzip -qq celeba_gan/data.zip -d celeba_gan
"""

def plot_image(dataset):
    import matplotlib.pyplot as plt
    for x in dataset:
        plt.axis("off")
        plt.imshow((x.numpy() * 255).astype("int32")[0])
        break

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis = 0)
        ## real images = 0, fake images = 1
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis = 0)
        labels *= 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))     

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(f"generated_img_{epoch:03d}_{i}.png")


def get_discriminator():
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    return discriminator

def get_generator(latent_dim):
    generator = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name = "generator",
    )

    return generator


if __name__ == "__main__":
    if Path("celeba_gan_dataset").is_dir():
        print("Loading dataset...")
        dataset = tf.data.experimental.load("celeba_gan_dataset")
    else:
        print("Creating dataset...")
        dataset = keras.utils.image_dataset_from_directory(
            "celeba_gan",
            label_mode=None,
            image_size=(64, 64),
            batch_size=32,
            smart_resize=True,
        )
        tf.data.experimental.save(dataset, "celeba_gan_dataset")
    
    if mixed_precision:
        dataset = dataset.map(lambda x: tf.cast((x / 255.), dtype="float16"))
    else:
        dataset = dataset.map(lambda x: x / 255.)
    print(f"Dataset: {dataset}")

    epochs = 100
    latent_dim = 128

    gan = GAN(
        discriminator=get_discriminator(),
        generator=get_generator(latent_dim),
        latent_dim=latent_dim,
    )

    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )


    callbacks = [
        keras.callbacks.ModelCheckpoint("gan.keras"),
        GANMonitor(num_img=10, latent_dim=latent_dim)
    ]

    gan.fit(dataset, epochs=epochs, callbacks=callbacks)
