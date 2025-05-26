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

def fit_and_plot(filepath, target_wavelengths):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    create_directory(output_folder)
    df = load_and_clean(filepath)
    plot_spectra(df, base_name, base_name)
    
    plot_dir = os.path.join(output_folder, base_name, "plots")
    create_directory(plot_dir)

    fit_params_list = []
    decay_constants_dict = {}

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
            ax.plot(x_vals, y_fit_single, 'g--', label=f"Single Exp Fit\n$R^2$={r2_single:.3f}")
            fit_params_list.append({
                "Spectrum": base_name,
                "Wavelength (nm)": target_wavelength,
                "Model": "Single",
                "A": popt_single[0],
                "k": popt_single[1],
                "C": popt_single[2],
                "R²": r2_single
            })
        else:
            try:
                popt, _ = curve_fit(double_exp, x_vals, y_vals, maxfev=10000, method='trf')
                y_fit = double_exp(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                x_fine = np.linspace(min(x_vals), max(x_vals), 100)
                y_fine = double_exp(x_fine, *popt)
                ax.plot(x_fine, y_fine, 'r--', label=f"Double Exp Fit\n$R^2$={r2:.3f}")

                decay_constants_dict[target_wavelength] = (popt[1], popt[3])
                fit_params_list.append({
                    "Spectrum": base_name,
                    "Wavelength (nm)": target_wavelength,
                    "A1": f"{popt[0]:.3e}",
                    "k1": f"{popt[1]:.3e}",
                    "A2": f"{popt[2]:.3e}",
                    "k2": f"{popt[3]:.3e}",
                    "C": f"{popt[4]:.3e}",
                    "R²": f"{r2:.3e}"
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

    pd.DataFrame(fit_params_list).to_csv(os.path.join(output_folder, base_name, "Fit_Params.csv"), index=False)

    # Implement comparison logic using k-values and returning the comparison dataframe
    if 400 in decay_constants_dict and 514 in decay_constants_dict:
        k2_400 = decay_constants_dict[400][1]
        k1_514 = decay_constants_dict[514][0]
        comparison_result = pd.DataFrame([{
            "Wavelength Transition": "400nm -> 514nm",
            "Previous k2": f"{k2_400:.3e}",
            "New k1": f"{k1_514:.3e}",
            "Difference": f"{(k1_514 - k2_400):.3e}"
        }])
        
        comparison_result.to_csv(os.path.join(output_folder, base_name, "Decay_Comparison.csv"), index=False)
        return comparison_result
    else:
        return None

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
    plt.xlim(200,700)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(f"Rescaled Spectrum — {label}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(rescaled_dir, f"Rescaled_Spectrum_{label}.png"), dpi=300)
    plt.close()
