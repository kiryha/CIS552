"""
CIS552 Assignment
Lab 3: Matplotlib, Seaborn, SciPy
"""

import matplotlib.pyplot as plt
import seaborn as sns


def task_1():

    # Coordinates
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]

    # Creating the scatter plot
    plt.scatter(x, y, color='red', marker='s')

    # Setting the range for x and y axes
    plt.xlim(0, 6)
    plt.ylim(0, 20)

    # Displaying the plot
    plt.show()


task_1()