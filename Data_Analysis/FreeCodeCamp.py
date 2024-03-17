import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from scipy import stats

def calculate(list):
    if len(list) < 9:
        raise ValueError("List must contain nine numbers.")

    matrix = np.array(list).reshape(3, 3)

    mean = [[], [], ]
    max = [[], [], ]
    min = [[], [], ]
    std = [[], [], ]
    var = [[], [], ]
    sum = [[], [], ]

    for i in range(len(matrix)):
        mean[1].append(np.mean(matrix[i, :]))
        max[1].append(np.max(matrix[i, :]))
        min[1].append(np.min(matrix[i, :]))
        std[1].append(np.std(matrix[i, :]))
        var[1].append(np.var(matrix[i, :]))
        sum[1].append(np.sum(matrix[i, :]))

    for j in range(len(matrix)):
        mean[0].append(np.mean(matrix[:, j]))
        max[0].append(np.max(matrix[:, j]))
        min[0].append(np.min(matrix[:, j]))
        std[0].append(np.std(matrix[:, j]))
        var[0].append(np.var(matrix[:, j]))
        sum[0].append(np.sum(matrix[:, j]))

    mean.append(np.mean(list))
    max.append(np.max(list))
    min.append(np.min(list))
    std.append(np.std(list))
    var.append(np.var(list))
    sum.append(np.sum(list))

    dic = {
        'mean': mean,
        'variance': var,
        'standard deviation': std,
        'max': max,
        'min': min,
        'sum': sum
    }

    print(dic)


def demographic():
    data = pd.read_csv('data.csv', sep=';')
    # How many people of each race are represented in this dataset?
    # This should be a Pandas series with race names as the index labels.
    race_count = data['race'].value_counts()

    # What is the average age of men?
    average_age_men = np.round(data['age'].mean(), decimals=1)
    print(average_age_men)

    # What is the percentage of people who have a Bachelor's degree?
    edu_count = data['education'].value_counts()
    bachelors_percent = np.round((edu_count.Bachelors / len(data) * 100), decimals=1)
    print(bachelors_percent)

    # What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50K?
    higher_edu = (len(data[((data['education'] == 'Bachelors') & (data['salary'] == '>50K')) |
                           (data['education'] == 'Masters') & (data['salary'] == '>50K') |
                           (data['education'] == 'Doctorate') & (data['salary'] == '>50K')])) / len(data)
    print(np.round(higher_edu * 100, decimals=1))

    # What percentage of people without advanced education make more than 50K?
    lower_edu = np.round(((len(data) - (higher_edu * len(data))) / len(data) * 100), decimals=1)
    print(lower_edu)

    # What is the minimum number of hours a person works per week?
    min_h = min(data['hours-per-week'])
    print(min_h)

    # What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?
    perc_h = np.round((len(data[(data['hours-per-week'] == min_h) & (data['salary'] == '>50K')]) / len(data)) * 100,
                      decimals=1)
    print(perc_h)

    # What country has the highest percentage of people that earn >50K and what is that percentage?
    country = data['native-country'][data['salary'] == '>50K'].value_counts().index[0]
    perc_c = np.round((data['native-country'][data['salary'] == '>50K'].value_counts().values[0] / len(data)) * 100,
                      decimals=1)
    print(country, perc_c)

    # Identify the most popular occupation for those who earn >50K in India.
    occupation = data['occupation'][(data['native-country'] == 'India') & (data['salary'] == '>50K')].value_counts().index[0]
    print(occupation)

def medical():
    data = pd.read_csv('medical_examination.csv', sep=';')

    # Add an overweight column to the data. To determine if a person is overweight,
    # first calculate their BMI by dividing their weight in kilograms by the square of their height in meters.
    # If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.
    # data['weight'] = data['weight']/2.205  # get kg, not necessary
    data['height'] = data['height']/100  # get m
    bmi = data['weight']/np.square(data['height'])
    data['bmi'] = bmi
    data['overweight'] = [1 if data['bmi'][i] > 25 else 0 for i in data['bmi'].index]

    # Normalize the data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1,
    # make the value 0. If the value is more than 1, make the value 1.
    data[['cholesterol', 'gluc']] = data[['cholesterol', 'gluc']].replace({1: 0, 2: 1, 3: 1})

    # Convert the data into long format and create a chart that shows the value counts of the categorical features
    # using seaborn's catplot(). The dataset should be split by 'Cardio' so there is one chart for each cardio value.
    # The chart should look like examples/Figure_1.png.

    def catplot(df):
        df_melted = pd.melt(df, id_vars=['cardio'],
                     value_vars=['active', 'alco', 'overweight', 'smoke', 'gluc', 'cholesterol'],
                            var_name='CategoricalFeature')

        fig = sns.catplot(df_melted, x='CategoricalFeature', hue='value', kind='count', col='cardio', aspect=1.2, height=4)
        fig.set_axis_labels("variable", "total")
        fig.savefig('catplot.png')
        plt.close()

        return fig.fig

    catplot(df=data)

    # clean the data
    d_s = data[data['ap_lo'] > data['ap_hi']].index
    height = data[(data['height'] < data['height'].quantile(0.025)) | (data['height'] > data['height'].quantile(0.975))].index
    weight = data[(data['weight'] < data['weight'].quantile(0.025)) | (data['weight'] > data['weight'].quantile(0.975))].index
    idx_list = list(d_s) + list(height) + list(weight)
    rem_idx = list(set(idx_list))
    data = data.drop(index=rem_idx, axis=0)

    def cor_matrix(df):
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, mask=mask, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        fig.savefig('correlation_matrix.png')

    cor_matrix(df=data)

def time_series_visualization():
    data = pd.read_csv('pageviews.csv', sep=';', index_col='date', parse_dates=True)

    # Clean the data by filtering out days when the page views were in the top 2.5% of the dataset
    # or bottom 2.5% of the dataset.
    data = data[(data['value'] >= data['value'].quantile(0.025)) & (data['value'] <= data['value'].quantile(0.975))]

    # Create a draw_line_plot function that uses Matplotlib to draw a line chart similar to "examples/Figure_1.png".
    # The title should be Daily freeCodeCamp Forum Page Views 5/2016-12/2019. The label on the x axis should be Date
    # and the label on the y axis should be Page Views.

    def draw_line_plot(df):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
        ax.set_xlabel('Date')
        ax.set_ylabel('Page Views')
        ax.plot(df, 'r')
        plt.show()
        plt.close()

    draw_line_plot(df=data)

    # Create a draw_bar_plot function that draws a bar chart similar to "examples/Figure_2.png".
    # It should show average daily page views for each month grouped by year.
    # The legend should show month labels and have a title of Months.
    # On the chart, the label on the x axis should be Years and the label on the y axis should be Average Page Views.

    def draw_bar_plot(df):
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        df_copy = df_copy.groupby(['year', 'month']).mean().reset_index()
        df_copy = df_copy.sort_values(by=['year', 'month'])
        df_copy['month'] = [calendar.month_name[i] for i in df_copy['month']]


        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_copy, x='year', y='value', hue='month', ax=ax, palette='Set1')
        ax.set_ylabel('Average views')
        plt.show()

    draw_bar_plot(df=data)

    # Create a draw_box_plot function that uses Seaborn to draw two adjacent box plots similar to
    # "examples/Figure_3.png". These box plots should show how the values are distributed within a given year or month
    # and how it compares over time. The title of the first chart should be Year-wise Box Plot (Trend) and the title of
    # the second chart should be Month-wise Box Plot (Seasonality). Make sure the month labels on bottom start at Jan
    # and the x and y axis are labeled correctly. The boilerplate includes commands to prepare the data.

    def draw_box_plot(df):
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        df_copy = df_copy.sort_values(by='month')
        df_copy['month'] = [calendar.month_name[i] for i in df_copy['month']]

        fig, axes = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)
        sns.boxplot(data=df_copy, x='year', y='value', ax=axes[0], hue='year', legend=False)
        sns.boxplot(data=df_copy, x='month', y='value', ax=axes[1], hue='month', legend=False)
        axes[0].set_ylabel('Page views')
        axes[1].set_ylabel('Page views')
        axes[0].set_xlabel('Year')
        axes[1].set_xlabel('Month')
        plt.show()

    draw_box_plot(df=data)


def sea_level_pred():
    df = pd.read_csv('sea_level.csv', sep=';')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x=df['Year'], y=df['CSIRO Adjusted Sea Level'])
    ax.set_xlim(1890, 2060)

    line_years = np.arange(1890, 2051, 1)
    years_ext = np.arange(2000, 2051, 1)

    res_all = stats.linregress(x=df['Year'], y=df['CSIRO Adjusted Sea Level'])
    res_2000 = stats.linregress(x=df.query('Year >= 2000')['Year'], y=df.query('Year >= 2000')['CSIRO Adjusted Sea Level'])

    pred_all = [res_all.slope * xi + res_all.intercept for xi in line_years]
    pred_2000 = [res_2000.slope * xi + res_2000.intercept for xi in years_ext]

    plt.plot(line_years, pred_all, color='red')
    plt.plot(years_ext, pred_2000, color='blue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (inches)')
    ax.set_title('Rise in sea level')
    ax.set_ylim(-5, 25)
    fig.savefig('sea_level.png')

if __name__ == '__main__':
    # list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # calculate(list)
    # demographic()
    # medical()
    # time_series_visualization()
    sea_level_pred()