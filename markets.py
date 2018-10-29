import tensorflow as tf
import numpy as np
from dataset import create

googl = create('GOOGL')
msft = create('MSFT')
aapl = create('AAPL')

# data = np.concatenate((aapl, googl, msft))
data = googl

feed_size = 10

print(len(data))

def gen():
  size = feed_size + 1
  for i in range(len(data) - size):
    window = data[i:i + size]
    
    input = window[:,0]
    input = [(input[i+1] - input[i]) / input[i] for i in range(len(input) - 1)]

    last = window[size - 1]
    last_open = last[0]
    last_close = last[3]
    change = (last_close - last_open) / last_open
    if change > 0:
      label = [1]
    else:
      label = [0]

    yield input, label, [last_open, last_close, change]

length = len(data)
dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([feed_size]), tf.TensorShape([1]), tf.TensorShape([3])))
# dataset = dataset.shuffle(length)
testing_length = int(length * 0.1)
# testing_length = 20
training_length = length - testing_length

training_dataset = dataset.take(training_length)
training_dataset = training_dataset.shuffle(training_length)
training_dataset = training_dataset.repeat()
training_dataset = training_dataset.batch(32)

testing_dataset = dataset.skip(training_length)
testing_dataset = testing_dataset.repeat()
testing_dataset = testing_dataset.batch(testing_length)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)

training_iterator = training_dataset.make_one_shot_iterator()
testing_iterator = testing_dataset.make_one_shot_iterator()

input, label, meta = iterator.get_next()

hidden_width = 10
hidden_depth = 10

hidden = input
for i in range(hidden_depth):
  dense = tf.layers.dense(inputs=hidden, units=hidden_width, activation=tf.tanh)
  concat = tf.concat([dense, hidden], 1)
  hidden = concat

output = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(label, output)
train_step = tf.train.AdagradOptimizer(1e-2).minimize(loss)
 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

training_handle = sess.run(training_iterator.string_handle())
testing_handle = sess.run(testing_iterator.string_handle())

for i in range(100000):
  train_step.run(feed_dict={handle: training_handle})

  if i % 1000 == 0:
    training_loss = loss.eval(feed_dict={handle: training_handle})
    testing_loss, b, c, d = sess.run([loss, label, output, meta], feed_dict={handle: testing_handle})
    e = np.concatenate((b, c, d), axis=1)

    u = 0
    v = 0
    w = 0
    for [label0, output0, open0, close0, change0] in e:
      u = u + change0

      if output0 > 0.5:
        v = v + change0

      if label0 > 0.5:
        w = w + change0

    u = u * 100
    v = v * 100
    w = w * 100
    d = v - u

    print("Step %d, training loss: %f testing loss: %f yield: %f ours: %f best: %f difference: %f"%(i, training_loss, testing_loss, u, v, w, d))
