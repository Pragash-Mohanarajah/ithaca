# %% [markdown]
# ##### Copyright 2023 The TensorFlow Authors.

# %%
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Import a JAX model using JAX2TF

# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/jax2tf"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/jax2tf.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/jax2tf.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/jax2tf.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# %% [markdown]
# This notebook provides a complete, runnable example of creating a model using [JAX](https://jax.readthedocs.io/en/latest/) and bringing it into TensorFlow to continue training. This is made possible by [JAX2TF](https://github.com/google/jax/tree/main/jax/experimental/jax2tf), a lightweight API that provides a pathway from the JAX ecosystem to the TensorFlow ecosystem. 
# 
# JAX is a high-performance array computing library. To create the model, this notebook uses [Flax](https://flax.readthedocs.io/en/latest/), a neural network library for JAX. To train it, it uses [Optax](https://optax.readthedocs.io), an optimization library for JAX.
# 
# If you're a researcher using JAX, JAX2TF gives you a path to production using TensorFlow's proven tools.
# 
# There are many ways this can be useful, here are just a few:
# 
# * Inference: Taking a model written for JAX and deploying it either on a server using TF Serving, on-device using TFLite, or on the web using TensorFlow.js. 
# 
# * Fine-tuning: Taking a model that was trained using JAX, you can bring its components to TF using JAX2TF, and continue training it in TensorFlow with your existing training data and setup.
# 
# * Fusion: Combining parts of models that were trained using JAX with those trained using TensorFlow, for maximum flexibility.
# 
# The key to enabling this kind of interoperation between JAX and TensorFlow is `jax2tf.convert`, which takes in model components created on top of JAX (your loss function, prediction function, etc) and creates equivalent representations of them as TensorFlow functions, which can then be exported as a TensorFlow SavedModel.

# %% [markdown]
# ## Setup
# 

# %%
import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax
import os
from matplotlib import pyplot as plt
from jax.experimental import jax2tf
from threading import Lock # Only used in the visualization utility.
from functools import partial

# %%
# Needed for TensorFlow and JAX to coexist in GPU memory.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized.
    print(e)

# %%
#@title Visualization utilities

plt.rcParams["figure.figsize"] = (20,8)

# The utility for displaying training and validation curves.
def display_train_curves(loss, avg_loss, eval_loss, eval_accuracy, epochs, steps_per_epochs, ignore_first_n=10):

  ignore_first_n_epochs = int(ignore_first_n/steps_per_epochs)

  # The losses.
  ax = plt.subplot(121)
  if loss is not None:
    x = np.arange(len(loss)) / steps_per_epochs #* epochs
    ax.plot(x, loss)
  ax.plot(range(1, epochs+1), avg_loss, "-o", linewidth=3)
  ax.plot(range(1, epochs+1), eval_loss, "-o", linewidth=3)
  ax.set_title('Loss')
  ax.set_ylabel('loss')
  ax.set_xlabel('epoch')
  if loss is not None:
    ax.set_ylim(0, np.max(loss[ignore_first_n:]))
    ax.legend(['train', 'avg train', 'eval'])
  else:
    ymin = np.min(avg_loss[ignore_first_n_epochs:])
    ymax = np.max(avg_loss[ignore_first_n_epochs:])
    ax.set_ylim(ymin-(ymax-ymin)/10, ymax+(ymax-ymin)/10)
    ax.legend(['avg train', 'eval'])

  # The accuracy.
  ax = plt.subplot(122)
  ax.set_title('Eval Accuracy')
  ax.set_ylabel('accuracy')
  ax.set_xlabel('epoch')
  ymin = np.min(eval_accuracy[ignore_first_n_epochs:])
  ymax = np.max(eval_accuracy[ignore_first_n_epochs:])
  ax.set_ylim(ymin-(ymax-ymin)/10, ymax+(ymax-ymin)/10)
  ax.plot(range(1, epochs+1), eval_accuracy, "-o", linewidth=3)

class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(reset=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bar
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # `()`: to get the iterator from the generator.
        self.header_printed = False
        self.msg = msg
        self.size = size
        self.lock = Lock()

    def step(self, reset=False):
        with self.lock:
            if reset:
                self.__init__(self.maxi, self.size, self.msg)
            if not self.header_printed:
                self.__print_header()
            next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k
            # Keep yielding the last result if there are too many steps.
            while True:
              yield k

        return print_progress

# %% [markdown]
# ## Download and prepare the MNIST dataset

# %%
(x_train, train_labels), (x_test, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
train_data = train_data.map(lambda x,y: (tf.expand_dims(tf.cast(x, tf.float32)/255.0, axis=-1),
                                         tf.one_hot(y, depth=10)))

BATCH_SIZE = 256
train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)
train_data = train_data.cache()
train_data = train_data.shuffle(5000, reshuffle_each_iteration=True)

test_data = tf.data.Dataset.from_tensor_slices((x_test, test_labels))
test_data = test_data.map(lambda x,y: (tf.expand_dims(tf.cast(x, tf.float32)/255.0, axis=-1),
                                         tf.one_hot(y, depth=10)))
test_data = test_data.batch(10000)
test_data = test_data.cache()

(one_batch, one_batch_labels) = next(iter(train_data)) # just one batch
(all_test_data, all_test_labels) = next(iter(test_data)) # all in one batch since batch size is 10000

# %% [markdown]
# ## Configure training
# This notebook will create and train a simple model for demonstration purposes.

# %%
# Training hyperparameters.
JAX_EPOCHS = 3
TF_EPOCHS = 7
STEPS_PER_EPOCH = len(train_labels)//BATCH_SIZE
LEARNING_RATE = 0.01
LEARNING_RATE_EXP_DECAY = 0.6

# The learning rate schedule for JAX (with Optax).
jlr_decay = optax.exponential_decay(LEARNING_RATE, transition_steps=STEPS_PER_EPOCH, decay_rate=LEARNING_RATE_EXP_DECAY, staircase=True)

# THe learning rate schedule for TensorFlow.
tflr_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=STEPS_PER_EPOCH, decay_rate=LEARNING_RATE_EXP_DECAY, staircase=True)

# %% [markdown]
# ## Create the model using Flax

# %%
class ConvModel(flax.linen.Module):

  @flax.linen.compact
  def __call__(self, x, train):
    x = flax.linen.Conv(features=12, kernel_size=(3,3), padding="SAME", use_bias=False)(x)
    x = flax.linen.BatchNorm(use_running_average=not train, use_scale=False, use_bias=True)(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = flax.linen.Dense(features=200, use_bias=True)(x)
    x = flax.linen.BatchNorm(use_running_average=not train, use_scale=False, use_bias=True)(x)
    x = flax.linen.Dropout(rate=0.3, deterministic=not train)(x)
    x = flax.linen.relu(x)
    x = flax.linen.Dense(features=10)(x)
    #x = flax.linen.log_softmax(x)
    return x

  # JAX differentiation requires a function `f(params, other_state, data, labels)` -> `loss` (as a single number).
  # `jax.grad` will differentiate it against the fist argument.
  # The user must split trainable and non-trainable variables into `params` and `other_state`.
  # Must pass a different RNG key each time for the dropout mask to be different.
  def loss(self, params, other_state, rng, data, labels, train):
    logits, batch_stats = self.apply({'params': params, **other_state},
                                     data,
                                     mutable=['batch_stats'],
                                     rngs={'dropout': rng},
                                     train=train)
    # The loss averaged across the batch dimension.
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    return loss, batch_stats

  def predict(self, state, data):
    logits = self.apply(state, data, train=False) # predict and accuracy disable dropout and use accumulated batch norm stats (train=False)
    probabilities = flax.linen.log_softmax(logits)
    return probabilities

  def accuracy(self, state, data, labels):
    probabilities = self.predict(state, data)
    predictions = jnp.argmax(probabilities, axis=-1)
    dense_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.equal(predictions, dense_labels).mean()
    return accuracy

# %% [markdown]
# ## Write the training step function

# %%
# The training step.
@partial(jax.jit, static_argnums=[0]) # this forces jax.jit to recompile for every new model
def train_step(model, state, optimizer_state, rng, data, labels):

  params = state['params'] # differentiate only against 'params' which represents trainable variables
  state.pop('params')
  other_state = state # non-trainable variables
  # state.pop('params')
  (loss, batch_stats), grads = jax.value_and_grad(model.loss, has_aux=True)(params, other_state, rng, data, labels, train=True)

  updates, optimizer_state = optimizer.update(grads, optimizer_state)
  params = optax.apply_updates(params, updates)
  new_state = state.copy()
  new_state.update({**batch_stats, 'params': params})

  rng, _ = jax.random.split(rng)

  return new_state, optimizer_state, rng, loss

# %% [markdown]
# ## Write the training loop

# %%
def train(model, state, optimizer_state, train_data, epochs, losses, avg_losses, eval_losses, eval_accuracies):
  p = Progress(STEPS_PER_EPOCH)
  rng = jax.random.PRNGKey(0)
  for epoch in range(epochs):

    # This is where the learning rate schedule state is stored in the optimizer state.
    optimizer_step = optimizer_state[1].count

    # Run an epoch of training.
    for step, (data, labels) in enumerate(train_data):
      p.step(reset=(step==0))
      state, optimizer_state, rng, loss = train_step(model, state, optimizer_state, rng, data.numpy(), labels.numpy())
      losses.append(loss)
    avg_loss = np.mean(losses[-step:])
    avg_losses.append(avg_loss)

    # Run one epoch of evals (10,000 test images in a single batch).
    params = state['params']
    state.pop('params')
    other_state = state
    state['params'] = params
    # Gotcha: must discard modified batch_stats here
    eval_loss, _ = model.loss(params, other_state, rng, all_test_data.numpy(), all_test_labels.numpy(), train=False)
    eval_losses.append(eval_loss)
    eval_accuracy = model.accuracy(state, all_test_data.numpy(), all_test_labels.numpy())
    eval_accuracies.append(eval_accuracy)

    print("\nEpoch", epoch, "train loss:", avg_loss, "eval loss:", eval_loss, "eval accuracy", eval_accuracy, "lr:", jlr_decay(optimizer_step))

  return state, optimizer_state

# %% [markdown]
# ## Create the model and the optimizer (with Optax)

# %%
# The model.
model = ConvModel()
state = model.init({'params':jax.random.PRNGKey(0), 'dropout':jax.random.PRNGKey(0)}, one_batch, train=True) # Flax allows a separate RNG for "dropout"

# The optimizer.
optimizer = optax.adam(learning_rate=jlr_decay) # Gotcha: it does not seem to be possible to pass just a callable as LR, must be an Optax Schedule
optimizer_state = optimizer.init(state['params'])

losses=[]
avg_losses=[]
eval_losses=[]
eval_accuracies=[]

# %% [markdown]
# ## Train the model

# %%
new_state, new_optimizer_state = train(model, state, optimizer_state, train_data, JAX_EPOCHS+TF_EPOCHS, losses, avg_losses, eval_losses, eval_accuracies)

# %%
display_train_curves(losses, avg_losses, eval_losses, eval_accuracies, len(eval_losses), STEPS_PER_EPOCH, ignore_first_n=1*STEPS_PER_EPOCH)

# %% [markdown]
# ## Partially train the model
# 
# You will continue training the model in TensorFlow shortly.

# %%
model = ConvModel()
state = model.init({'params':jax.random.PRNGKey(0), 'dropout':jax.random.PRNGKey(0)}, one_batch, train=True) # Flax allows a separate RNG for "dropout"

# The optimizer.
optimizer = optax.adam(learning_rate=jlr_decay) # LR must be an Optax LR Schedule
optimizer_state = optimizer.init(state['params'])

losses, avg_losses, eval_losses, eval_accuracies = [], [], [], []

# %%
state, optimizer_state = train(model, state, optimizer_state, train_data, JAX_EPOCHS, losses, avg_losses, eval_losses, eval_accuracies)

# %%
display_train_curves(losses, avg_losses, eval_losses, eval_accuracies, len(eval_losses), STEPS_PER_EPOCH, ignore_first_n=1*STEPS_PER_EPOCH)

# %% [markdown]
# ## Save just enough for inference
# 
# If your goal is to deploy your JAX model (so you can run inference using `model.predict()`), simply exporting it to [SavedModel](https://www.tensorflow.org/guide/saved_model) is sufficient. This section demonstrates how to accomplish that.

# %%
# Test data with a different batch size to test polymorphic shapes.
x, y = next(iter(train_data.unbatch().batch(13)))

m = tf.Module()
# Wrap the JAX state in `tf.Variable` (needed when calling the converted JAX function.
state_vars = tf.nest.map_structure(tf.Variable, state)
# Keep the wrapped state as flat list (needed in TensorFlow fine-tuning).
m.vars = tf.nest.flatten(state_vars)
# Convert the desired JAX function (`model.predict`).
predict_fn = jax2tf.convert(model.predict, polymorphic_shapes=["...", "(b, 28, 28, 1)"])
# Wrap the converted function in `tf.function` with the correct `tf.TensorSpec` (necessary for dynamic shapes to work).
@tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32)])
def predict(data):
    return predict_fn(state_vars, data)
m.predict = predict
tf.saved_model.save(m, "./")

# %%
# Test the converted function.
print("Converted function predictions:", np.argmax(m.predict(x).numpy(), axis=-1))
# Reload the model.
reloaded_model = tf.saved_model.load("./")
# Test the reloaded converted function (the result should be the same).
print("Reloaded  function predictions:", np.argmax(reloaded_model.predict(x).numpy(), axis=-1))

# %% [markdown]
# ## Save everything
# If your goal is a comprehensive export (useful if you're planning on brining the model into TensorFlow for fine-tuning, fusion, etc), this section demonstrates how to save the model so you can access methods including:
# 
#  - model.predict
#  - model.accuracy
#  - model.loss (including train=True/False bool, RNG for dropout and BatchNorm state updates)

# %%
from collections import abc

def _fix_frozen(d):
  """Changes any mappings (e.g. frozendict) back to dict."""
  if isinstance(d, list):
    return [_fix_frozen(v) for v in d]
  elif isinstance(d, tuple):
    return tuple(_fix_frozen(v) for v in d)
  elif not isinstance(d, abc.Mapping):
    return d
  d = dict(d)
  for k, v in d.items():
    d[k] = _fix_frozen(v)
  return d

# %%
class TFModel(tf.Module):
  def __init__(self, state, model):
    super().__init__()

    # Special care needed for the train=True/False parameter in the loss
    @jax.jit
    def loss_with_train_bool(state, rng, data, labels, train):
      other_state, params = state.pop('params')
      loss, batch_stats = jax.lax.cond(train,
                                       lambda state, data, labels: model.loss(params, other_state, rng, data, labels, train=True),
                                       lambda state, data, labels: model.loss(params, other_state, rng, data, labels, train=False),
                                       state, data, labels)
      # must use JAX to split the RNG, therefore, must do it in a @jax.jit function
      new_rng, _ = jax.random.split(rng)
      return loss, batch_stats, new_rng

    self.state_vars = tf.nest.map_structure(tf.Variable, state)
    self.vars = tf.nest.flatten(self.state_vars)
    self.jax_rng = tf.Variable(jax.random.PRNGKey(0))

    self.loss_fn = jax2tf.convert(loss_with_train_bool, polymorphic_shapes=["...", "...", "(b, 28, 28, 1)", "(b, 10)", "..."])
    self.accuracy_fn = jax2tf.convert(model.accuracy, polymorphic_shapes=["...", "(b, 28, 28, 1)", "(b, 10)"])
    self.predict_fn = jax2tf.convert(model.predict, polymorphic_shapes=["...", "(b, 28, 28, 1)"])

  # Must specify TensorSpec manually for variable batch size to work
  @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32)])
  def predict(self, data):
    # Make sure the TfModel.predict function implicitly use self.state_vars and not the JAX state directly
    # otherwise, all model weights would be embedded in the TF graph as constants.
    return self.predict_fn(self.state_vars, data)

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
               autograph=False)
  def train_loss(self, data, labels):
      loss, batch_stats, new_rng = self.loss_fn(self.state_vars, self.jax_rng, data, labels, True)
      # update batch norm stats
      flat_vars = tf.nest.flatten(self.state_vars['batch_stats'])
      flat_values = tf.nest.flatten(batch_stats['batch_stats'])
      for var, val in zip(flat_vars, flat_values):
        var.assign(val)
      # update RNG
      self.jax_rng.assign(new_rng)
      return loss

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
               autograph=False)
  def eval_loss(self, data, labels):
      loss, batch_stats, new_rng = self.loss_fn(self.state_vars, self.jax_rng, data, labels, False)
      return loss

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                                tf.TensorSpec(shape=(None, 10), dtype=tf.float32)],
               autograph=False)
  def accuracy(self, data, labels):
    return self.accuracy_fn(self.state_vars, data, labels)

# %%
# Instantiate the model.
tf_model = TFModel(state, model)

# Save the model.
tf.saved_model.save(tf_model, "./")

# %% [markdown]
# ## Reload the model

# %%
reloaded_model = tf.saved_model.load("./")

# Test if it works and that the batch size is indeed variable.
x,y = next(iter(train_data.unbatch().batch(13)))
print(np.argmax(reloaded_model.predict(x).numpy(), axis=-1))
x,y = next(iter(train_data.unbatch().batch(20)))
print(np.argmax(reloaded_model.predict(x).numpy(), axis=-1))

print(reloaded_model.accuracy(one_batch, one_batch_labels))
print(reloaded_model.accuracy(all_test_data, all_test_labels))

# %% [markdown]
# ## Continue training the converted JAX model in TensorFlow

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=tflr_decay)

# Set the iteration step for the learning rate to resume from where it left off in JAX.
optimizer.iterations.assign(len(eval_losses)*STEPS_PER_EPOCH)

p = Progress(STEPS_PER_EPOCH)

for epoch in range(JAX_EPOCHS, JAX_EPOCHS+TF_EPOCHS):

  # This is where the learning rate schedule state is stored in the optimizer state.
  optimizer_step = optimizer.iterations

  for step, (data, labels) in enumerate(train_data):
    p.step(reset=(step==0))
    with tf.GradientTape() as tape:
      #loss = reloaded_model.loss(data, labels, True)
      loss = reloaded_model.train_loss(data, labels)
      grads = tape.gradient(loss, reloaded_model.vars)
      optimizer.apply_gradients(zip(grads, reloaded_model.vars))
      losses.append(loss)
  avg_loss = np.mean(losses[-step:])
  avg_losses.append(avg_loss)

  eval_loss = reloaded_model.eval_loss(all_test_data.numpy(), all_test_labels.numpy()).numpy()
  eval_losses.append(eval_loss)
  eval_accuracy = reloaded_model.accuracy(all_test_data.numpy(), all_test_labels.numpy()).numpy()
  eval_accuracies.append(eval_accuracy)

  print("\nEpoch", epoch, "train loss:", avg_loss, "eval loss:", eval_loss, "eval accuracy", eval_accuracy, "lr:", tflr_decay(optimizer.iterations).numpy())

# %%
display_train_curves(losses, avg_losses, eval_losses, eval_accuracies, len(eval_losses), STEPS_PER_EPOCH, ignore_first_n=2*STEPS_PER_EPOCH)

# The loss takes a hit when the training restarts, but does not go back to random levels.
# This is likely caused by the optimizer momentum being reinitialized.

# %% [markdown]
# ## Next steps
# You can learn more about [JAX](https://jax.readthedocs.io/en/latest/index.html) and [Flax](https://flax.readthedocs.io/en/latest) on their documentation websites which contain detailed guides and examples. If you're new to JAX, be sure to explore the [JAX 101 tutorials](https://jax.readthedocs.io/en/latest/jax-101/index.html), and check out the [Flax quickstart](https://flax.readthedocs.io/en/latest/getting_started.html). To learn more about converting JAX models to TensorFlow format, check out the [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf) utility on GitHub. If you're interested in converting JAX models to run in the browser with TensorFlow.js, visit [JAX on the Web with TensorFlow.js](https://blog.tensorflow.org/2022/08/jax-on-web-with-tensorflowjs.html). If you'd like to prepare JAX models to run in TensorFLow Lite, visit the [JAX Model Conversion For TFLite](https://www.tensorflow.org/lite/examples/jax_conversion/overview) guide.


