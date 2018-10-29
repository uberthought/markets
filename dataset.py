import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime

def create(name):
    fileName = 'data/' + name

    with open(fileName,'rb') as file:
        data = pickle.load(file)

    data_list = list(reversed(list(data.items())))

    # names = np.repeat(name, len(data))
    # dates = [datetime.strptime(k[0], '%Y-%m-%d').timestamp() for k in data_list]
    opens = [float(k[1]['1. open']) for k in data_list]
    highs = [float(k[1]['2. high']) for k in data_list]
    lows = [float(k[1]['3. low']) for k in data_list]
    closes = [float(k[1]['4. close']) for k in data_list]

    return np.stack((opens, highs, lows, closes), axis=1)

# if __name__ == "__main__":
#     dataset = create('GOOGL')
#     print(dataset.output_types)
#     print(dataset.output_shapes)
