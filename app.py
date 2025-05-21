import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import workflow

st.set_page_config(page_title="UV-Vis Analyzer", layout="wide")
logo_image_path = "download.png"
st.image(logo_image_path, width=300)
st.title("UV-Vis Spectrum Analyzer")

uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file is not None:
  file_path = os.path.join("uploaded", uploaded_file.name)
  os.makedirs("uploaded", exist_ok=True)
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.success(f"File uploaded: {uploaded_file.name}")

  workflow.target_wavelengths = [400, 514]

  with st.spinner("Running analysis..."):
    workflow.fit_and_plot(file_path, workflow.target_wavelengths)
    st.success("Analysis complete!")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(base_name, "plots")

    if os.path.exists(output_dir):
      st.subheader("Spectrum Plots")
      for img_file in sorted(os.listdir(output_dir)):
         if img_file.startswith("Full") or img_file.startswith("Rescaled"):
          st.image(os.path.join(output_dir, img_file), caption=img_file, use_column_width=True)

      st.subheader("Fit Plots")
      for img_file in sorted(os.listdir(output_dir)):
        if img_file.startswith("Fit_"):
          st.image(os.path.join(output_dir, img_file), caption=img_file, use_column_width=True)

      csv_path = os.path.join(base_name, "Fit_Params.csv")
      if os.path.exists(csv_path):
        st.subheader("Fitted Parameters")
        df = pd.read_csv(csv_path)
        st.dataframe(df)
        st.download_button("Download Fit_Params.csv", df.to_csv(index=False),
        file_name="Fit_Params.csv")
