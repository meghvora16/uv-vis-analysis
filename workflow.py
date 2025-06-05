import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def triple_exp(t, A1, k1, A2, k2, A3, k3, C):
    t = np.array(t, dtype=float)
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + A3 * np.exp(-k3 * t) + C

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

def format_to_exponential(value):
    return f"{value:.3e}"

def fit_and_plot(filepath, target_wavelengths, exp_type):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    create_directory(output_folder)
    df = load_and_clean(filepath)
    plot_spectra(df, base_name, base_name)

    plot_dir = os.path.join(output_folder, base_name, "plots")
    create_directory(plot_dir)

    fit_params_list = []

    for target_wavelength in target_wavelengths:
        idx = (df.iloc[:, 0] - target_wavelength).abs().idxmin()
        y_vals = df.iloc[idx, 1:].to_numpy()
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_vals, y_vals, color="black", label="Data")

        # Fit according to the selected exponential type
        # Single Exponential fitting section in fit_and_plot function
        if exp_type == "Single Exponential":
            try:
                popt, _ = curve_fit(single_exp, x_vals, y_vals, maxfev=10000)
                y_fit = single_exp(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                ax.plot(x_vals, y_fit, 'g--', label=f"Single Exp Fit\n$R^2$={r2:.3f}")
                fit_params_list.append({
                "Spectrum": base_name,
                "Wavelength (nm)": target_wavelength,
                "Model": "Single",
                "A": format_to_exponential(popt[0]),
                "k": format_to_exponential(popt[1]),
                "C": format_to_exponential(popt[2]),
                "R²": format_to_exponential(r2)
                })
            except RuntimeError:
                st.error(f"Single exponential fit failed for wavelength {target_wavelength} nm.")

        elif exp_type == "Double Exponential":
            try:
                popt, _ = curve_fit(double_exp, x_vals, y_vals, maxfev=10000)
                y_fit = double_exp(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                ax.plot(x_vals, y_fit, 'r--', label=f"Double Exp Fit\n$R^2$={r2:.3f}")
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "Model": "Double",
                    "A1": format_to_exponential(popt[0]),
                    "k1": format_to_exponential(popt[1]),
                    "A2": format_to_exponential(popt[2]),
                    "k2": format_to_exponential(popt[3]),
                    "C": format_to_exponential(popt[4]),
                    "R²": format_to_exponential(r2)
                })
            except RuntimeError:
                st.error(f"Double exponential fit failed for wavelength {target_wavelength} nm.")

        elif exp_type == "Triple Exponential":
            try:
                popt, _ = curve_fit(triple_exp, x_vals, y_vals, maxfev=10000)
                y_fit = triple_exp(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                ax.plot(x_vals, y_fit, 'b--', label=f"Triple Exp Fit\n$R^2$={r2:.3f}")
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "Model": "Triple",
                    "A1": format_to_exponential(popt[0]),
                    "k1": format_to_exponential(popt[1]),
                    "A2": format_to_exponential(popt[2]),
                    "k2": format_to_exponential(popt[3]),
                    "A3": format_to_exponential(popt[4]),
                    "k3": format_to_exponential(popt[5]),
                    "C": format_to_exponential(popt[6]),
                    "R²": format_to_exponential(r2)
                })
            except RuntimeError:
                st.error(f"Triple exponential fit failed for wavelength {target_wavelength} nm.")

        ax.set_title(f"{base_name} — Fits at {target_wavelength} nm")
        ax.set_xlabel("Spectrum Index")
        ax.set_ylabel("Absorbance")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Fit_{target_wavelength}nm.png"))
        plt.close()

    fit_params_df = pd.DataFrame(fit_params_list, dtype=str)
    fit_params_df.to_csv(os.path.join(output_folder, base_name, "Fit_Params.csv"), index=False)

    return fit_params_df

def plot_spectra(df, filename, label):
    wavelengths = df.iloc[:, 0].to_numpy()
    rescaled_dir = os.path.join(output_folder, filename, "plots")
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

    plt.ylim(0, 1)
    plt.xlim(200, 700)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"Rescaled Spectrum — {label}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(rescaled_dir, f"Rescaled_Spectrum_{label}.png"), dpi=300)
    plt.close()
