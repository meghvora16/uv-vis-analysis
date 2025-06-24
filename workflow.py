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
    return A * np.exp(-k * t) + C

def double_exp(t, A1, k1, A2, k2, C):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C

def triple_exp(t, A1, k1, A2, k2, A3, k3, C):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + A3 * np.exp(-k3 * t) + C

def load_and_clean(filepath):
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin1')
        if df.empty:
            raise ValueError(f"No data found in file: {filepath}")
        for column in df.columns:
            df[column] = pd.to_numeric(df[column].str.replace(',', '.'), errors='coerce')
    except Exception as e:
        print(f"Error loading the file {filepath}: {e}")
        return None
    return df

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_to_exponential(value):
    try:
        return f"{value:.3e}"
    except:
        return "NaN"

def fit_model(x_vals, y_vals, model_fn, initial_guess, bounds, dense_x):
    popt, _ = curve_fit(model_fn, x_vals, y_vals, p0=initial_guess, bounds=bounds, maxfev=10000)
    y_fit = model_fn(dense_x, *popt)
    fitted_y = model_fn(x_vals, *popt)
    r2 = r2_score(y_vals, fitted_y)
    return popt, y_fit, r2

def fit_and_plot(filepath, target_wavelengths, exp_type):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    plot_dir = os.path.join(output_folder, base_name, "plots")
    create_directory(plot_dir)

    df = load_and_clean(filepath)
    if df is None:
        return pd.DataFrame()

    plot_spectra(df, base_name)
    fit_params_list = []

    for target_wavelength in target_wavelengths:
        idx = (df.iloc[:, 0] - target_wavelength).abs().idxmin()
        y_vals = df.iloc[idx, 1:].to_numpy(dtype=float)
        if len(y_vals) == 0:
            print(f"No y-values found for wavelength {target_wavelength} nm in {base_name}.")
            continue
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float) * 360
        x_dense = np.linspace(x_vals.min(), x_vals.max(), 500)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_vals, y_vals, color="black", label="Data")

        try:
            if exp_type == "Single Exponential":
                initial_guess = [max(y_vals), 0.001, min(y_vals)]
                bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                popt, y_fit, r2 = fit_model(x_vals, y_vals, single_exp, initial_guess, bounds, x_dense)
                half_life = np.log(2) / popt[1] if popt[1] > 0 else np.nan
                ax.plot(x_dense, y_fit, 'g--', label=f"Single Exp Fit\n$R^2$={r2:.3f}\n$t_{{1/2}}$={half_life:.2f}s")
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "Model": "Single",
                    "A": format_to_exponential(popt[0]),
                    "k": format_to_exponential(popt[1]),
                    "C": format_to_exponential(popt[2]),
                    "R²": format_to_exponential(r2),
                    "Half-life (s)": format_to_exponential(half_life)
                })
            elif exp_type == "Double Exponential":
                initial_guess = [max(y_vals)/2, 0.001, max(y_vals)/2, 0.0001, min(y_vals)]
                bounds = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
                popt, y_fit, r2 = fit_model(x_vals, y_vals, double_exp, initial_guess, bounds, x_dense)
                half_life1 = np.log(2) / popt[1] if popt[1] > 0 else np.nan
                half_life2 = np.log(2) / popt[3] if popt[3] > 0 else np.nan
                ax.plot(x_dense, y_fit, 'r--', label=f"Double Exp Fit\n$R^2$={r2:.3f}\n$t_{{1/2,1}}$={half_life1:.2f}s\n$t_{{1/2,2}}$={half_life2:.2f}s")
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "Model": "Double",
                    "A1": format_to_exponential(popt[0]),
                    "k1": format_to_exponential(popt[1]),
                    "A2": format_to_exponential(popt[2]),
                    "k2": format_to_exponential(popt[3]),
                    "C": format_to_exponential(popt[4]),
                    "R²": format_to_exponential(r2),
                    "Half-life1 (s)": format_to_exponential(half_life1),
                    "Half-life2 (s)": format_to_exponential(half_life2)
                })
            elif exp_type == "Triple Exponential":
                initial_guess = [max(y_vals)/3, 0.001, max(y_vals)/3, 0.0001, max(y_vals)/3, 0.00001, min(y_vals)]
                bounds = ([0, 0, 0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                popt, y_fit, r2 = fit_model(x_vals, y_vals, triple_exp, initial_guess, bounds, x_dense)
                half_life1 = np.log(2) / popt[1] if popt[1] > 0 else np.nan
                half_life2 = np.log(2) / popt[3] if popt[3] > 0 else np.nan
                half_life3 = np.log(2) / popt[5] if popt[5] > 0 else np.nan
                ax.plot(x_dense, y_fit, 'b--', label=f"Triple Exp Fit\n$R^2$={r2:.3f}\n$t_{{1/2,1}}$={half_life1:.2f}s\n$t_{{1/2,2}}$={half_life2:.2f}s\n$t_{{1/2,3}}$={half_life3:.2f}s")
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
                    "R²": format_to_exponential(r2),
                    "Half-life1 (s)": format_to_exponential(half_life1),
                    "Half-life2 (s)": format_to_exponential(half_life2),
                    "Half-life3 (s)": format_to_exponential(half_life3)
                })
            else:
                print(f"Unknown exp_type: {exp_type}")
                continue

        except RuntimeError as re:
            print(f"{exp_type} fit failed for wavelength {target_wavelength} nm on spectrum {base_name}: {re}")
        except Exception as err:
            print(f"An error occurred during fitting for wavelength {target_wavelength} nm on spectrum {base_name}: {err}")

        ax.set_title(f"{base_name} — Fits at {target_wavelength} nm")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absorbance")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Fit_{target_wavelength}nm.png"))
        plt.close()

    fit_params_df = pd.DataFrame(fit_params_list, dtype=str)
    fit_params_file = os.path.join(output_folder, base_name, "Fit_Params.csv")
    fit_params_df.to_csv(fit_params_file, index=False)
    print(f"Fit parameters saved: {fit_params_file}")
    return fit_params_df

def plot_spectra(df, label):
    wavelengths = df.iloc[:, 0].to_numpy()
    plot_dir = os.path.join(output_folder, label, "plots")
    create_directory(plot_dir)

    # Full Spectrum Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, df.shape[1]):
        ax.plot(wavelengths, df.iloc[:, i], label=f"Spectrum {i}", alpha=0.7)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Full Spectrum - {label}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    full_spec_file = os.path.join(plot_dir, f"Full_Spectrum_{label}.png")
    plt.savefig(full_spec_file, dpi=300)
    plt.close()

    # Rescaled Spectrum Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, df.shape[1]):
        ax.plot(wavelengths, df.iloc[:, i], label=f"Spectrum {i}", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(200, 700)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Rescaled Spectrum — {label}")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    rescaled_spec_file = os.path.join(plot_dir, f"Rescaled_Spectrum_{label}.png")
    plt.savefig(rescaled_spec_file, dpi=300)
    plt.close()
