"""
CIS552 Assignment
Lab 2: NumPy and Pandas
"""

import numpy as np
import pandas as pd


def task_1():

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # print(a.shape)

    return a


def task_2():

    a = task_1()
    b = a.reshape(3, 2)
    # print(b.shape)

    return b


def task_3():

    b = task_2()
    c = b.transpose()

    print(c.shape)


def task_4():

    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6], [7, 8]])

    dot_product = np.dot(array_1, array_2)

    print(dot_product)


def task_5():

    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6], [7, 8]])

    multiply = array_1 * array_2

    print(multiply)


def task_6():

    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6], [7, 8]])

    result_axis_0 = np.concatenate((array_1, array_2), axis=0)

    print(result_axis_0)
    print("Shape:", result_axis_0.shape)

    result_axis_1 = np.concatenate((array_1, array_2), axis=1)

    print(result_axis_1)
    print("Shape:", result_axis_1.shape)


def task_7():

    array_1 = np.array([[1, 2], [3, 4]])

    multiply = array_1 * 3

    print(multiply)


def task_8():

    array_1 = np.array([[1, 2], [3, 4]])

    array_2 = array_1.copy()
    array_2[0, 0] = 0

    print(array_1)
    print(array_2)


def task_9():

    array_1 = np.array([[1, 2], [3, 4]])

    array_2 = array_1.view()
    array_2[0, 0] = 0

    print(array_1)
    print(array_2)


def task_10():

    array_1 = np.array([[1, 2], [3, 4]])

    np.save('test.np', array_1)
    array_2 = np.load('test.np.npy')

    print(array_1)
    print(array_2)


def task_11():

    array = np.array([[1, 2, 3], [4, 5, 6]])
    transposed_array = array.T

    print(transposed_array)


def task_12():

    array_3 = np.array([[1, 2, 3], [4, 5, 6]])
    array_4 = np.array([1, 2])

    addition = array_3 + array_4[:, np.newaxis]

    print(addition)


def task_13():

    input_list = ['a', 'b', 'c', 'd']
    series = pd.Series(input_list)

    print(series)


def task_14():

    data_list = [2, 4, 6, 8]
    index_list = ['a', 'b', 'c', 'd']

    series = pd.Series(data_list, index=index_list)

    print(series)


def task_15_1():

    data_list = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    series = pd.Series(data_list)

    print(series)

    filtered_series = series[series >= 7]

    print(filtered_series)


def task_15_2():

    data_list = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    series = pd.Series(data_list)

    print(series)

    booleanized_series = series >= 7

    print(booleanized_series)


def task_15_3():

    data_list = [2, 4, 6, 8, np.nan]
    series = pd.Series(data_list)

    print(series)

    multiplied_series = series * 5

    print(multiplied_series)


def task_16():

    data = {'state': ['FL', 'FL', 'GA', 'GA', 'GA'],
            'year': [2010, 2011, 2008, 2010, 2011],
            'pop': [18.8, 19.1, 9.7, 9.7, 9.8]}

    dataframe = pd.DataFrame(data)

    print(dataframe)


def task_17():

    data = {'state': ['FL', 'FL', 'GA', 'GA', 'GA'],
            'year': [2010, 2011, 2008, 2010, 2011],
            'pop': [18.8, 19.1, 9.7, 9.7, 9.8]}

    dataframe = pd.DataFrame(data)

    print(dataframe.describe())


def task_18():

    dataframe = pd.read_csv('tips.csv')

    print(dataframe.head(3))


def task_20():

    dataframe = pd.read_csv('tips.csv')
    normalized_cross_table = pd.crosstab(dataframe['day'], dataframe['size'], normalize='columns')

    print(normalized_cross_table)


task_20()
