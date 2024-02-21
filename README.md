# Annotation QC Tool :rocket:
A simple, fast and user friendly web application for simplifying and enhancing the Quality Control (QC) process for annotated datasets!

![QCTOOL](assets/Annotation_Tool_Recording.gif)

---
# Prerequisites

The current prerequisites are as follows:
1. OBB annotated dataset (.txt) files.
   - Format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
2. 4 - 8GB RAM 

---
# Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/azista-image-processing/Annotation_QC_Tool.git
    ```

2. **Go to the repository**
    ```bash
    cd Annotation_QC_Tool/
    ```

3. **Create an environment**
   - Conda environment 
        ```bash
        conda create -n qcTool -y && conda activate qcTool
        ```

   - Python environment (for Mac & Linux)
        ```bash
        python -m venv qcTool && cd && cd qcTool/bin && source ./activate
        ```

    - Python environment (for Windows)
        ```bash
        python -m venv qcTool && cd && cd qcTool/Scripts && source activate
        ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**
    ```bash
    streamlit run qcTool.py
    ```

---
# To - Do
- [x] Moved To-Dos to GitHub Issues

