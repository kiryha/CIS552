"""
CIS552 Assignment
Lab 3: Matplotlib, Seaborn, SciPy
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def task_1():

    # Data
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]

    # Creating the scatter plot
    plt.scatter(x, y, color='red', marker='s')

    # Setting the range for x and y axes
    plt.xlim(0, 6)
    plt.ylim(0, 20)

    # Displaying the plot
    plt.show()


def task_2():

    # Data
    x = np.array([0, 1, 2, 3, 4, 5])
    y = x  # Straight line
    y_curve1 = x ** 2  # Curve 1
    y_curve2 = x ** 3  # Curve 2

    # Creating the plot
    plt.figure()

    # Plotting curves
    plt.plot(x, y, 'r--', label='Straight line')  # Red dashed line
    plt.plot(x, y_curve1, 'bo', label='Curve 1')  # Blue circles
    plt.plot(x, y_curve2, 'g^', label='Curve 2')  # Green triangles

    # Adding a legend
    plt.legend()

    # Displaying the plot
    plt.show()


def task_3():

    # Generating random data points
    mu, sigma = 100, 10  # mean and standard deviation
    x = mu + sigma * np.random.randn(1000)  # 1000 random data points

    # Creating the histogram
    plt.hist(x, bins=20, alpha=0.75, edgecolor='black', linewidth=1.5)

    # Adding a background grid
    plt.grid(True)

    # Displaying the histogram
    plt.show()


def task_4():

    # Generating three sets of random data points
    mu, sigma = 100, 10
    x = mu + sigma * np.random.randn(1000, 3)  # 1000 random data points for each of the 3 sets

    # Colors for the histograms
    colors = ['red', 'green', 'blue']

    # Creating the histograms
    plt.figure()
    for i in range(3):
        plt.hist(x[:, i], bins=20, alpha=0.75, color=colors[i], edgecolor='black', linewidth=1.5, label=f'Set {i + 1}')

    # Adding a background grid
    plt.grid(True)

    # Adding a legend
    plt.legend()

    # Displaying the histogram
    plt.show()


def task_5():

    # Read the CSV file into a DataFrame
    df = pd.read_csv('BostonHousing.csv')

    # Draw a boxplot for the 'dis' column
    sns.boxplot(x=df['dis'])

    # Display the plot
    plt.show()


def task_6():

    # Reading the CSV file
    df = pd.read_csv('BostonHousing.csv')

    # Melting the DataFrame
    melted_df = pd.melt(df)

    # Creating boxplots for all variables
    sns.boxplot(x='variable', y='value', data=melted_df)

    # Rotating x labels for better readability
    plt.xticks(rotation=90)

    # Displaying the plot
    plt.show()


def task_7():

    # Reading the CSV file
    df = pd.read_csv('BostonHousing.csv')

    # Calculating the pairwise correlation
    correlation_matrix = df.corr()

    # Creating a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # Displaying the plot
    plt.show()


def task_8():

    # Reading the CSV file
    df = pd.read_csv('BostonHousing.csv')

    # Creating a pairplot
    sns.pairplot(df)

    # Displaying the plot
    plt.show()


task_8()
