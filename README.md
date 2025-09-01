# Trans-CNN: A Transition-Aware Approach to Sleep Apnea Detection using a Lightweight Global-Local CNN
This repository contains the official implementation for the paper "Trans-CNN: A Transition-Aware Approach to Sleep Apnea Detection using a Lightweight Global-Local CNN". We propose a deep learning model to detect sleep apnea events on a minute-by-minute basis using single-lead ECG signals from the Apnea-ECG Database.

## Project Structure

The repository is organized as follows:

```
.
├── data
│   ├── data.pkl                   # The preprocessed feature file (needs to be downloaded)
│   └── additional-information.txt # Subject-level ground truth (from the original dataset)
│
├── data_loader.py                 # Loads and prepares the preprocessed data for the model
├── model.py                       # Defines the TransCNN architecture and custom layers
├── train.py                       # Main script for training and evaluating the model
├── analysis.py                    # Script for subject-level analysis and generating final results
├── requirements.txt               # Required Python libraries
├── LICENSE                        # License
└── README.md                      # This file
```           

## Setup and Installation

### 1. Prerequisites
- Python 3.7.12
- TensorFlow 2.10.0
- Other dependencies listed in requirements.txt.
### 2. Clone the Repository

```bash
git clone https://github.com/ly1228140504/TransCNN-Sleep-Apnea.git
cd TransCNN_Sleep_Apnea
```

### 3. Create a Virtual Environment and Install Dependencies
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on Windows)
venv\Scripts\activate

# Activate it (on macOS/Linux)
source venv/
bin
/activate

# Install the required packages
pip install -r requirements.txt
```

## How to Run the Full PipelineHow

Follow these steps sequentially to train the model and generate the final results.

### ➡️ Step 1: Configure Paths

Before running the scripts, ensure the file paths are correctly set. Open the following files and check that the `base_dir` variable at the top points to your data folder:
- `data_loader.py`
- `train.py`
- `analysis.py`

The variable should be set like this:
```python
base_dir = '.\data'
```
### ➡️ Step 2: Model Training and Evaluation
This script loads the preprocessed data.pkl, trains the model, and saves the minute-by-minute predictions.

```bash
python train.py
```
- Input: The data.pkl file you downloaded.
- Output:
  - `Result.xlsx`: Contains the minute-by-minute performance metrics (Accuracy, Sensitivity, Specificity, F1-score) for each of the 10 runs.
  - `Method.xlsx`: Contains the raw prediction scores for every minute of the test set across all runs.
  - Model weights (`.keras` files) for the best epoch of each run.

### ➡️ Step 3: Subject-Level Analysis
This final script aggregates the minute-by-minute predictions to a per-subject level, calculates the Apnea-Hypopnea Index (AHI), and computes the final clinical evaluation metrics presented in our paper.

```bash
python analysis.py
```
- Input: `Method.xlsx` from the training step and the `additional-information.txt` from the original dataset.
- Output:
  - `Table 2.csv`: A CSV file containing the final subject-level results, including Accuracy, Sensitivity, Specificity, AUC, and Correlation.
  - The final results will also be printed to the console.

## File Descriptions

- data_loader.py: Handles loading the data.pkl feature file, including interpolation and splitting into train/validation/test sets.
- model.py: Defines the Keras/TensorFlow implementation of our TransCNN model, including the custom attention layers.
- train.py: The primary script for orchestrating the model training and evaluation loop over 10 runs. It saves model weights and prediction results.
- analysis.py: Performs the final subject-level aggregation and metric calculation, translating minute-level predictions into clinically relevant statistics.

## Data Availability
### A Note on Data and Preprocessing
- As our manuscript is currently undergoing peer review, we have temporarily withheld the data preprocessing scripts and the final data.pkl file to maintain confidentiality during the academic review process.
- However, we have already open-sourced the complete code for our model architecture (model.py), the training and evaluation pipeline (train.py), and the final subject-level analysis (analysis.py). We hope this provides valuable insight into our methodology for the community to reference and build upon.
- We are deeply committed to open and reproducible research. Upon the successful publication of our paper, we promise to release the entire preprocessing pipeline and the final data.pkl file in this repository.
- For academic reviewers: We would be pleased to provide private access to the complete codebase and data to facilitate a thorough evaluation of our work. Please feel free to contact the corresponding author to make arrangements.
- We sincerely appreciate your interest in our project and thank you for your patience and understanding.


## Citation

If you use this code in your research, please consider citing our paper:

```bibtex
@article{your_citation_key,
  author    = {Author, A. and Author, B.},
  title     = {Trans-CNN: A Transition-Aware Approach to Sleep Apnea Detection using a Lightweight Global-Local CNN},
  journal   = {Journal Name},
  year      = {2025},
  volume    = {XX},
  number    = {Y},
  pages     = {ZZZ--ZZZ}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.