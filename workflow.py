import streamlit as st
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the output folder path upfront
output_folder = "Combined_Fits"

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
    
def single_exp(t, A, k, C): 
    t = np.array(t, dtype=float)
    return A * np.exp(-k * t) + C
    
def double_exp(t, A1, k1, A2, k2, C):
    t = np.array(t, dtype=float)
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C

def load_and_clean(filepath):
    df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin1')
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '.')
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_days(file_paths):
    days = []
    for label in file_paths:
        match = re.search(r'\d+', label)
        if match:
            days.append(int(match.group()))
    return np.array(days, dtype=float)

file_paths = {}

create_directory(output_folder)

days = extract_days(file_paths)
time_seconds = days * 86400
target_wavelengths = [400, 514]

def plot_spectra(df, filename, label):
    wavelengths = df.iloc[:, 0].to_numpy()
    rescaled_dir = os.path.join(filename, "plots")
    create_directory(rescaled_dir)

    plt.figure(figsize=(10, 5))
    num_columns = df.shape[1]
    for i in range(1, num_columns):
        plt.plot(wavelengths, df.iloc[:, i], label=f"Spectrum {i}", alpha=0.7)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"Full Spectrum - {label}")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(rescaled_dir, f"Full_Spectrum_{label}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for i in range(1, num_columns):
        plt.plot(wavelengths, df.iloc[:, i], label=f"Spectrum {i}", alpha=0.7)

    plt.ylim(0.2, 1)
    plt.xlim(200,700)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"Rescaled Spectrum — {label}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(rescaled_dir, f"Rescaled_Spectrum_{label}.png"), dpi=300)
    plt.close()

def fit_and_plot(filepath, target_wavelengths):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    create_directory(base_name)

    df = load_and_clean(filepath)
    plot_spectra(df, base_name, base_name)

    plot_dir = os.path.join(base_name, "plots")
    create_directory(plot_dir)

    fit_params_list = []

    for target_wavelength in target_wavelengths:
        idx = (df.iloc[:, 0] - target_wavelength).abs().idxmin()
        y_vals = df.iloc[idx, 1:].to_numpy()
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_vals, y_vals, color="black", label="Data")

        try:
            popt_single, _ = curve_fit(single_exp, x_vals, y_vals, maxfev=10000)
            y_fit_single = single_exp(x_vals, *popt_single)
            r2_single = r2_score(y_vals, y_fit_single)
        except RuntimeError:
            r2_single = -np.inf
            popt_single = None
        R2_THRESHOLD = 1.0
        if popt_single is not None and r2_single >= R2_THRESHOLD:
            half_life = np.log(2) / popt_single[1] if popt_single[1] != 0 else np.nan
            ax.plot(x_vals, y_fit_single, 'g--', label=f"Single Exp Fit\n$R^2$={r2_single:.3f}")
            fit_params_list.append({
                "Spectrum": base_name,
                "Wavelength (nm)": target_wavelength,
                "Model": "Single",
                "A": popt_single[0],
                "k": popt_single[1],
                "half_life_sec": half_life,
                "C": popt_single[2],
                "R²": r2_single
            })
        else:
            try:
                popt, _ = curve_fit(double_exp, x_vals, y_vals, maxfev=10000, method='trf')
                y_fit = double_exp(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                half_life1 = (np.log(2) / popt[1]) * 86400 if popt[1] != 0 else np.nan
                half_life2 = (np.log(2) / popt[3]) * 86400 if popt[3] != 0 else np.nan
                x_fine = np.linspace(min(x_vals), max(x_vals), 100)
                y_fine = double_exp(x_fine, *popt)
                ax.plot(x_fine, y_fine, 'r--', label=f"Double Exp Fit\n$R^2$={r2:.3f}")
    
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "A1": popt[0], "k1": popt[1],
                    "half_life1_sec": half_life1,
                    "A2": popt[2], "k2": popt[3],
                    "half_life2_sec": half_life2,
                    "C": popt[4], "R²": r2
                })

            except RuntimeError:
                print(f"Fit failed for {base_name} at {target_wavelength} nm")

        ax.set_title(f"{base_name} — Fit at {target_wavelength} nm")
        ax.set_xlabel("Spectrum Index")
        ax.set_ylabel("Absorbance")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Fit_{target_wavelength}nm.png"))
        plt.close()

    pd.DataFrame(fit_params_list).to_csv(os.path.join(base_name, "Fit_Params.csv"), index=False)
    if 400 in decay_constants_dict and 514 in decay_constants_dict:
        k2_400 = decay_constants_dict[400][1]
        k1_514 = decay_constants_dict[514][0]
        comparison_result = {
            "Spectrum": base_name,
            "k2_400": k2_400,
            "k1_514": k1_514,
            "Difference": k1_514 - k2_400
        }

        pd.DataFrame([comparison_result]).to_csv(os.path.join(base_name, "Decay_Comparison.csv"), index=False)
