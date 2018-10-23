import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime

def create(name):
    fileName = 'data/' + name

    with open(fileName,'rb') as file:
        data = pickle.load(file)

    # names = np.repeat(name, len(data))
    # dates = [datetime.strptime(k[0], '%Y-%m-%d').timestamp() for k in data.items()]
    opens = [float(k[1]['1. open']) for k in data.items()]
    highs = [float(k[1]['2. high']) for k in data.items()]
    lows = [float(k[1]['3. low']) for k in data.items()]
    closes = [float(k[1]['4. close']) for k in data.items()]

    return np.stack((opens, highs, lows, closes), axis=1)
    # max = np.max(foo)
    # min = np.min(foo)
    # foo = (foo - min) / (max - min)
    # first = foo[0][0]
    # foo = foo - first

    # return foo
    # return tf.data.Dataset.from_tensor_slices(foo)

# if __name__ == "__main__":
#     dataset = create('GOOGL')
#     print(dataset.output_types)
#     print(dataset.output_shapes)
