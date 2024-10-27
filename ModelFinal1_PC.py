import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

country1 = pd.read_csv('russia.csv')
Country_1 = 'Russia'
country2 = pd.read_csv('canada.csv')
Country_2 = 'Canada'

for component in ['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production', 'Nuclear Production',
                  'Other Production',
                  'Industry', 'Transport', 'Households', 'Other', 'Agriculture', 'Commercial', 'Energy Imports', 
                  'Energy Exports', 'Total Energy Use', 'GDP']:
    country1[f'{component} per Capita'] = country1[component] / country1['Population']
    country2[f'{component} per Capita'] = country2[component] / country2['Population']

def forecast_energy(data, feature, years_to_predict=5):
    X = data['Year'].values.reshape(-1, 1)
    y = data[feature].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 1 + years_to_predict).reshape(-1, 1)
    future_values = model.predict(future_years)

    return future_years.flatten(), future_values

sns.set(style='whitegrid')

def create_subplot_group(components, title):
    num_components = len(components)
    rows = (num_components + 1) // 2
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    for idx, component in enumerate(components):
        row = idx // cols
        col = idx % cols

        axes[row, col].plot(country1['Year'], country1[f'{component} per Capita'], label=f'{Country_1} Historical', marker='o')
        axes[row, col].plot(country2['Year'], country2[f'{component} per Capita'], label=f'{Country_2} Historical', marker='s')

        future_years, future_values_country1 = forecast_energy(country1, f'{component} per Capita')
        _, future_values_country2 = forecast_energy(country2, f'{component} per Capita')
        axes[row, col].plot(future_years, future_values_country1, label=f'{Country_1} Forecast', linestyle='--')
        axes[row, col].plot(future_years, future_values_country2, label=f'{Country_2} Forecast', linestyle='--')

        axes[row, col].set_title(f'{component} per Capita')
        axes[row, col].set_xlabel('Year')
        axes[row, col].set_ylabel(f'{component} per Capita')
        axes[row, col].legend()
        axes[row, col].grid(True)

    if num_components % cols != 0:
        for empty_subplot in range(num_components, rows * cols):
            fig.delaxes(axes.flatten()[empty_subplot])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_3d_energy_gdp(data, country_name):
    energy_components = ['Industry per Capita', 'Transport per Capita', 'Households per Capita', 'Other per Capita', 'Agriculture per Capita', 'Commercial per Capita']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'x', 'D', 'P']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for component, color, marker in zip(energy_components, colors, markers):
        ax.scatter(data['Year'], data['GDP per Capita'], data[component],
                   label=f'{component}', color=color, marker=marker, s=50)

    ax.set_xlabel('Year')
    ax.set_ylabel('GDP per Capita')
    ax.set_zlabel('Energy Consumption per Capita')
    ax.set_title(f'3D Plot of GDP, Year, and Energy Consumption for {country_name}')

    ax.legend()

    plt.tight_layout()
    plt.show()

plot_3d_energy_gdp(country1, Country_1)
plot_3d_energy_gdp(country2, Country_2)

def plot_3d_energy_gdp_comparison(data1, data2, country_name1, country_name2):
    energy_components = ['Industry per Capita', 'Transport per Capita', 'Households per Capita', 'Other per Capita', 'Agriculture per Capita', 'Commercial per Capita']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    markers = ['o', 's', '^', 'x', 'D', 'P']

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for component, color, marker in zip(energy_components, colors, markers):
        ax.plot(data1['Year'], data1['GDP per Capita'], data1[component],
                label=f'{country_name1} - {component}', color=color, marker=marker, linestyle='-', linewidth=1)

    for component, color, marker in zip(energy_components, colors, markers):
        ax.plot(data2['Year'], data2['GDP per Capita'], data2[component],
                label=f'{country_name2} - {component}', color=color, marker=marker, linestyle='--', linewidth=1)

    ax.set_xlabel('Year')
    ax.set_ylabel('GDP per Capita')
    ax.set_zlabel('Energy Consumption per Capita')
    ax.set_title(f'3D Comparison of GDP, Year, and Energy Consumption for {country_name1} and {country_name2}')

    ax.legend()

    plt.tight_layout()
    plt.show()

plot_3d_energy_gdp_comparison(country1, country2, Country_1, Country_2)

create_subplot_group(['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production', 'Nuclear Production', 
                      'Other Production'],
                     'Energy Production per Capita Comparison 1')

create_subplot_group(['Industry', 'Transport', 'Households', 'Other', 'Agriculture', 'Commercial'], 
                     'Energy Consumption per Capita Comparison 2')

create_subplot_group(['Energy Imports', 'Energy Exports', 'Total Energy Use', 'GDP'], 
                     'Energy Imports, Exports, and Economic Indicators per Capita Comparison')
