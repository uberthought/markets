from alpha_vantage.timeseries import TimeSeries
import pickle


def collect(name):
    fileName = 'data/' + name
    try:
        with open(fileName,'rb') as file:
            data = pickle.load(file)
            print('cached data')
    except Exception as e:
        ts = TimeSeries(key='ALUW2LQ7UFWXU8RP')
        data, _ = ts.get_daily(name, outputsize='full')
        with open(fileName,'wb') as file:
            pickle.dump(data, file)
            file.close()
        print('live data')

    print(len(data))

collect('GOOGL')
collect('MSFT')
collect('AAPL')