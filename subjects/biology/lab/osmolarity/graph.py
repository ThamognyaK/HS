from typing import List
import numpy as np
import matplotlib.pyplot as plt

CONCENTRATION: np.ndarray = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])

raw_initial_mass: np.ndarray = np.array([[1.21, 1.12, 1.26, 1.19],
                                         [1.22, 1.02, 1.28, 1.08],
                                         [1.20, 1.14, 1.29, 1.20],
                                         [1.18, 1.08, 1.29, 1.17],
                                         [1.23, 1.11, 1.31, 1.08],
                                         [1.17, 1.13, 1.3, 1.15]])

raw_final_mass: np.ndarray = np.array([[1.24, 1.15, 1.32, 1.25],
                                       [1.24, 1.04, 1.32, 1.20],
                                       [1.15, 1.12, 1.24, 1.10],
                                       [1.07, 1.02, 1.21, 1.01],
                                       [1.05, 1.04, 1.19, 0.98],
                                       [1.00, 0.99, 1.11, 0.99]])

def calculate_mean(initial_mass: np.ndarray, final_mass: np.ndarray) -> np.ndarray:
    return np.mean(final_mass / initial_mass * 100, axis=1)

def calculate_sd(initial_mass: np.ndarray, final_mass: np.ndarray) -> np.ndarray:
    return np.std(final_mass / initial_mass * 100, axis=1, ddof=1)

def calculate_se(sd: np.ndarray) -> np.ndarray:
    return sd / np.sqrt(4)

def calculate_ci(se: np.ndarray) -> np.ndarray:
    return se * 1.96

def cleanse_data(data: np.ndarray) -> np.ndarray:
    cleaned_data = []
    for arr in data.T:  # Transpose the array to loop over the columns
        Q1 = np.quantile(arr, 0.25)
        Q3 = np.quantile(arr, 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_arr = np.where((arr < lower_bound) | (arr > upper_bound), np.nan, arr)
        cleaned_arr = np.where(np.isnan(cleaned_arr), np.nanmedian(cleaned_arr), cleaned_arr)
        cleaned_data.append(cleaned_arr)
    return np.array(cleaned_data).T

def plot(concentration: np.ndarray, mean: np.ndarray, ci: np.ndarray, title: str) -> None:
    plt.errorbar(concentration, mean, yerr=ci, fmt='o', capsize=5)
    plt.xlabel('Concentration of Sucrose Solution (M)')
    plt.ylabel('Percent Change in Average Mass (%)')
    plt.title(title)
    plt.show()

def plot_raw() -> None:
    raw_mean: np.ndarray = calculate_mean(raw_initial_mass, raw_final_mass)
    raw_sd: np.ndarray = calculate_sd(raw_initial_mass, raw_final_mass)
    raw_se: np.ndarray = calculate_se(raw_sd)
    raw_ci: np.ndarray = calculate_ci(raw_se)
    title: str = 'Percent Change in Average Mass (%) vs Concentration of Sucrose Solution (M)'
    plot(CONCENTRATION, raw_mean, raw_ci, title)

def plot_clean() -> None:
    cleaned_initial_mass = cleanse_data(raw_initial_mass)
    cleaned_final_mass = cleanse_data(raw_final_mass)
    mean: np.ndarray = calculate_mean(cleaned_initial_mass, cleaned_final_mass)
    sd: np.ndarray = calculate_sd(cleaned_initial_mass, cleaned_final_mass)
    se: np.ndarray = calculate_se(sd)
    ci: np.ndarray = calculate_ci(se)
    title: str = '(outliers removed) Percent Change in Average Mass (%) vs Concentration of Sucrose Solution (M)'
    plot(CONCENTRATION, mean, ci, title)

if __name__ == "__main__":
    # plot_raw()
    plot_clean()
