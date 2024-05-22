import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import time
import logging
import datetime


class SlotAttentionModule(layers.Layer):
    """Slot Attention module."""
    def __init__(self, num_slots, num_iters, slot_size, mlp_hidden_dim, eps=1e-8):
        super(SlotAttentionModule, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.slot_size = slot_size
        self.eps = eps
        self.mlp_hidden_dim = mlp_hidden_dim
        
        self.input_norm = layers.LayerNormalization()
        # random_normal used below - glorot_uniform in the original implementation
        # also trainable=True is used below - not there in the original implementation
        self.mu_slot = self.add_weight(shape=(1, 1, slot_size), initializer='random_normal', trainable=True, dtype=tf.float32)
        self.sigma_slot = self.add_weight(shape=(1, 1, slot_size), initializer='random_normal', trainable=True, dtype=tf.float32)
        self.k_proj = layers.Dense(slot_size, use_bias=False)
        self.q_proj = layers.Dense(slot_size, use_bias=False)
        self.v_proj = layers.Dense(slot_size, use_bias=False)
        self.slots_norm = layers.LayerNormalization()
        self.gru = layers.GRU(slot_size) # GRU to update the slots - GRUCell in the original implementation
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='relu'),
            layers.Dense(slot_size)
        ])
        self.mlp_norm = layers.LayerNormalization()
    
    def forward(self, x):
        """Forward pass."""
        inputs = self.input_norm(x)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        slots = self.mu_slot + tf.exp(self.sigma_slot) * tf.random.normal((tf.shape(inputs)[0], self.num_slots, self.slot_size))
        
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.slots_norm(slots)
            q = self.q_proj(slots)
            q *= tf.math.sqrt(tf.cast(self.slot_size, tf.float32))
            attn = tf.nn.softmax(tf.keras.backend.batch_dot(k, q, axes=-1), axis=-1)
            attn = (attn + self.eps) / tf.reduce_sum(attn, axis=-2, keepdims=True)
            updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
            slots, _ = self.gru(updates, initial_state=slots_prev) # GRUCell in the original implementation
            slots += self.mlp(self.mlp_norm(slots))
            
        return slots
        

def grid_embed(resolution):
    scope = [np.linspace(0.0, 1.0, num=x) for x in resolution]
    grid = np.meshgrid(*scope, indexing='ij', sparse=False)
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (resolution[0], resolution[1], -1))
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SlotAttentionNetwork(layers.Layer):
    """Slot Attention network."""
    def __init__(self, num_slots, num_iters, resolution):
        super(SlotAttentionNetwork, self).__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.resolution = resolution
        
        self.cnn_encoder = tf.keras.Sequential([
            layers.Conv2D(64, 5, padding='same', activation='relu'),
            layers.Conv2D(64, 5, padding='same', activation='relu'),
            layers.Conv2D(64, 5, padding='same', activation='relu'),
            layers.Conv2D(64, 5, padding='same', activation='relu')
        ])
        self.dense_encode = layers.Dense(64, use_bias=True)
        self.norm = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(64)
        ])
        self.slot_attention = SlotAttentionModule(num_slots, num_iters, 64, 128)
        self.decoder_size = (8, 8)
        self.dense_decode = layers.Dense(64, use_bias=True)
        self.cnn_decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 5, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 5, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 5, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 5, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, 5, 1, padding='same', activation='relu'),
            layers.Conv2DTranspose(4, 3, 1, padding='same', activation=None)
        ])
        
    def forward(self, input):
        """Forward pass."""
        x = self.cnn_encoder(input)
        x += self.dense_encode(grid_embed(self.resolution))
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[-1]))
        x = self.mlp(self.norm(x))
        slots = self.slot_attention(x)
        x = tf.tile(tf.reshape(slots, (-1, slots.shape[-1]))[:, None, None, :], (1, self.decoder_size[0], self.decoder_size[1], 1))
        x += self.dense_decode(grid_embed(self.decoder_size))
        x = self.cnn_decoder(x)
        unstacked = tf.reshape(x, (input.shape[0], -1) + x.shape.as_list()[1:])
        recons, masks = tf.split(unstacked, (3, 1), axis=-1)
        masks = tf.nn.sigmoid(masks, axis=1)
        combined = tf.reduce_sum(recons * masks, axis=1)
        
        return recons, masks, combined, slots


def build(num_slots, num_iters, resolution, batch_size, num_channels=3):
    """Build the model."""
    input = tf.keras.Input(list(resolution) + [num_channels], batch_size=batch_size)
    model = SlotAttentionNetwork(num_slots, num_iters, resolution)
    recons, masks, combined, slots = model(input)
    return tf.keras.Model(inputs=input, outputs=[recons, masks, combined, slots])


# hyperparameters
batch_size = 64
num_slots = 7
num_iters = 3
resolution = (128, 128)
learning_rate = 4e-4
num_train_steps = 500000
warmup_steps = 10000
decay_rate = 0.5
decay_steps = 100000
tf.random.set_seed(0)

# data iterator - to be modified
data_iter = iter(tf.data.Dataset.from_tensor_slices(np.random.randn(batch_size, *resolution, 3)).batch(batch_size))

optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)
model = build(num_slots, num_iters, resolution, batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

start = time.time()
for _ in range(num_train_steps):
    input = next(data_iter)
    if global_step < warmup_steps:
        lr = learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
        lr = learning_rate
    lr *= decay_rate ** (tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32))
    optimizer.lr = lr
    with tf.GradientTape() as tape:
        recons, masks, combined, slots = model(input["image"], training=True)
        loss = tf.reduce_mean(tf.math.squared_difference(input["image"], combined))
        del recons, masks, slots
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    global_step.assign_add(1)
    if global_step % 100 == 0:
        logging.info("Step: %s, Loss: %.6f, Time: %s", global_step.numpy(), loss, datetime.timedelta(seconds=time.time() - start))

model.save_weights("model_a2.pt")



