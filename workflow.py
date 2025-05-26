import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

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

file_paths = {
    "3d": "ExxonMobil_Schaffler_LV-AU_150°C_3d.txt",
    "10d": "ExxonMobil_Schaffler_LV-AU_150°C_10d.txt",
    "15d": "ExxonMobil_Schaffler_LV-AU_150°C_15d.txt",
    "21d": "ExxonMobil_Schaffler_LV-AU_150°C_21d.txt"
}

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
            popt, _ = curve_fit(double_exp, x_vals, y_vals, maxfev=10000, method='trf')
            y_fit = double_exp(x_vals, *popt)
            r2 = r2_score(y_vals, y_fit)
            half_life1 = np.log(2) / popt[1] if popt[1] != 0 else np.nan
            half_life2 = np.log(2) / popt[3] if popt[3] != 0 else np.nan
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

    pd.DataFrame(fit_params_list).to_csv(os.path.join(base_name,
"Fit_Params.csv"), index=False)

def fit_across_files(file_paths, target_wavelengths, output_folder):
    create_directory(output_folder)
    output_txt = os.path.join(output_folder, "double_exp_fit_parameters.txt")
    with open(output_txt, "w") as f:
        f.write("Double Exponential Fit Parameters (A1, k1, A2, k2, C)\n\n")

    for wavelength in target_wavelengths:
        plt.figure(figsize=(8, 5))

        for spectrum_idx in range(1, 11):
            absorbance_vals = []

            for label, path in file_paths.items():
                df = load_and_clean(path)
                idx = (df.iloc[:, 0] - wavelength).abs().idxmin()
                absorbance_vals.append(df.iloc[idx, spectrum_idx])

            absorbance_vals = np.array(absorbance_vals)

            try:
                popt, _ = curve_fit(double_exp, days, absorbance_vals, maxfev=10000, method='trf')
                fit_curve = double_exp(days, *popt)

                plt.scatter(days, absorbance_vals, label=f"Spectrum {spectrum_idx}", alpha=0.6)
                plt.plot(days, fit_curve, linestyle='--', alpha=0.7)

                with open(output_txt, "a") as f:
                    f.write(f"Wavelength: {wavelength} nm | Spectrum: {spectrum_idx}\n")
                    f.write(f"Parameters: A1={popt[0]:.5f}, k1={popt[1]:.5f}, "
                            f"A2={popt[2]:.5f}, k2={popt[3]:.5f}, C={popt[4]:.5f}\n\n")

            except RuntimeError:
                print(f"Fit failed for Spectrum {spectrum_idx} at {wavelength} nm")
                with open(output_txt, "a") as f:
                    f.write(f"Wavelength: {wavelength} nm | Spectrum: {spectrum_idx} - Fit Failed\n\n")

        plt.xlabel("Days of Exposure")
        plt.ylabel("Absorbance")
        plt.title(f"Double Exponential Fit at {wavelength} nm (All Files)")
        plt.grid()
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"double_exp_fit_{wavelength}nm.png"), dpi=300)
        plt.close()

# Run
for filepath in file_paths.values():
    fit_and_plot(filepath, target_wavelengths)

fit_across_files(file_paths, target_wavelengths, output_folder="Combined_Fits")
