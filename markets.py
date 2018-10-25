import tensorflow as tf
from dataset import create

data = create('GOOGL')

print(len(data))

def gen():
  size = 6
  for i in range(len(data) - size):
    window = data[i:i + size]
    
    input = window[:,0]
    input = [10 * (input[i] - input[i+1]) / input[i] for i in range(len(input) - 1)]

    # label = [window[size - 1][3]]
    last = window[size - 1]
    if last[0] < last[3]:
      label = [1]
    else:
      label = [0]

    yield input, label

length = len(data)
dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32), (tf.TensorShape([5]), tf.TensorShape([1])))
dataset = dataset.shuffle(length)
testing_length = int(length * 0.1)

testing_dataset = dataset.take(testing_length)
testing_dataset = testing_dataset.repeat()
testing_dataset = testing_dataset.batch(testing_length)

training_dataset = dataset.skip(testing_length)
training_dataset = training_dataset.repeat()
training_dataset = training_dataset.batch(32)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)

training_iterator = training_dataset.make_one_shot_iterator()
testing_iterator = testing_dataset.make_one_shot_iterator()

input, label = iterator.get_next()

hidden1 = tf.layers.dense(inputs=input, units=5, activation=tf.tanh)
concat1 = tf.concat([input, hidden1], 1)
hidden2 = tf.layers.dense(inputs=concat1, units=5, activation=tf.tanh)
concat2 = tf.concat([concat1, hidden2], 1)
output = tf.layers.dense(inputs=concat2, units=1, activation=tf.sigmoid)

loss = tf.losses.mean_squared_error(label, output)
train_step = tf.train.AdagradOptimizer(1e-1).minimize(loss)
diff = tf.subtract(label, output)
 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

training_handle = sess.run(training_iterator.string_handle())
testing_handle = sess.run(testing_iterator.string_handle())

for i in range(100000):
  train_step.run(feed_dict={handle: training_handle})

  if i % 1000 == 0:
    training_loss = loss.eval(feed_dict={handle: training_handle})
    testing_loss = loss.eval(feed_dict={handle: testing_handle})
    print("Step %d, training loss: %f testing loss: %f"%(i, training_loss, testing_loss))

print(loss.eval(feed_dict={handle: training_handle}))
print(loss.eval(feed_dict={handle: testing_handle}))