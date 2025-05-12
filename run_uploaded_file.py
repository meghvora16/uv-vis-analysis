import sys
import workflow
import os
import subprocess

try:
    if len(sys.argv) != 2:
        raise ValueError("File path not passed.")

    uploaded_path = sys.argv[1]
    workflow.target_wavelengths = [400, 514]
    workflow.fit_and_plot(uploaded_path, workflow.target_wavelengths)
    base_name = os.path.splitext(os.path.basename(uploaded_path))[0]
    output_dir = os.path.abspath(base_name)

    with open("log.txt", "a") as log:
        log.write(f"Successfully processed: {uploaded_path}\n")

    if os.path.exists(output_dir):
        subprocess.run(["explorer", output_dir])

except Exception as e:
    with open("log.txt", "a") as log:
        log.write(f"ERROR: {str(e)}\n")