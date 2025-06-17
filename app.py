import streamlit as st
import pandas as pd
import os
import workflow
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit page configuration
st.set_page_config(page_title="UV-Vis Analyzer", layout="wide")
logo_image_path = "Schaeffler_Logo.png"
st.image(logo_image_path, width=200)
st.title("UV-Vis Spectrum Analyzer")

# Dropdown menu for selecting the exponential type
exp_type = st.selectbox(
    "Select Exponential Fitting Type",
    ["Please select an option", "Single Exponential", "Double Exponential", "Triple Exponential"]
)

# File uploader
uploaded_files = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"], accept_multiple_files=True)
save_dir = "uploaded"
os.makedirs(save_dir, exist_ok=True)

# Ensure target wavelengths are defined globally in workflow
workflow.target_wavelengths = [400, 514]

# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")

        # Run the analysis for this file
        with st.spinner(f"Running analysis on {uploaded_file.name}..."):
            # Pass the exponential type to `fit_and_plot`
            try:
                k_comparison_df = workflow.fit_and_plot(file_path, workflow.target_wavelengths, exp_type)
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                continue

        st.success(f"Analysis complete for {uploaded_file.name}!")

        # Define output directory for plots
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(workflow.output_folder, base_name, "plots")

        # Display spectrum plots if they exist
        if os.path.exists(output_dir):
            st.subheader("Spectrum Plots:")
            for img_file in sorted(os.listdir(output_dir)):
                if img_file.startswith("Full") or img_file.startswith("Rescaled"):
                    st.image(os.path.join(output_dir, img_file), caption=img_file, use_container_width=True)

            st.subheader("Fit Plots:")
            for img_file in sorted(os.listdir(output_dir)):
                if img_file.startswith("Fit_"):
                    st.image(os.path.join(output_dir, img_file), caption=img_file, use_container_width=True)

        # Display fitted parameters if they exist
        csv_path = os.path.join(workflow.output_folder, base_name, "Fit_Params.csv")
        if os.path.exists(csv_path):
            st.subheader("Fitted Parameters:")
            df = pd.read_csv(csv_path)
            st.dataframe(df)
            st.download_button(
                "Download Fit_Params.csv",
                df.to_csv(index=False),
                file_name=f"Fit_Params_{uploaded_file.name}.csv",
                key=f"download_btn_{uploaded_file.name}"
            )
