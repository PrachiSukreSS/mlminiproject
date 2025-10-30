# mlminiproject
# Music Genre Classification (K-Nearest Neighbors)

A small machine learning mini-project demonstrating music genre classification using a K-Nearest Neighbors (KNN) classifier. The project creates a synthetic dataset of audio features, preprocesses the data, trains and evaluates a KNN model, and visualizes results (confusion matrix, accuracy vs k, feature correlations, and 2D scatter of genres).

This repository contains a single script: `ML_Miniproject.py`, which generates the dataset and performs the full modeling and visualization pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [How the Code Works](#how-the-code-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Outputs and Visualizations](#outputs-and-visualizations)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This mini-project demonstrates a simple approach to music genre classification using K-Nearest Neighbors (KNN). It is intended as an educational example covering:

- Creating a synthetic dataset of audio features (danceability, energy, tempo, loudness, etc.).
- Preprocessing (train/test split, standard scaling).
- Model selection for KNN by testing a range of k values.
- Training the final model and reporting classification metrics.
- Visualizing performance and data relationships.

---

## Dataset

The dataset in `ML_Miniproject.py` is synthetically generated (no external data files):

- 200 samples
- Features:
  - Danceability (0–1)
  - Energy (0–1)
  - Tempo (BPM: 60–199)
  - Loudness (dB: -60 to 0)
  - Acousticness (0–1)
  - Instrumentalness (0–1)
  - Valence (0–1)
  - Duration_ms (150,000–300,000)
- Target labels (genres): Pop, Rock, Classical, Jazz (uniform random choice)

> Note: Because the dataset is randomly generated, model accuracy and plots will vary across runs. The script sets a random seed for reproducibility of each run (`np.random.seed(42)`).

---

## How the Code Works

High-level pipeline implemented in `ML_Miniproject.py`:

1. Generate synthetic dataset and create a pandas DataFrame.
2. Separate features (X) and target (y).
3. Split the data into training and testing sets (test_size=0.25, random_state=42).
4. Standardize features using `StandardScaler`.
5. Evaluate KNN for different values of k (k=1..14):
   - Train KNN on scaled training data.
   - Predict on scaled test data.
   - Measure accuracy for each k.
6. Select the best `k` that yields the highest test accuracy.
7. Train the final KNN model with the best `k` and print classification report and confusion matrix.
8. Produce visualizations:
   - Confusion matrix (heatmap)
   - Accuracy vs k (line plot)
   - Feature correlation heatmap
   - 2D scatter plot of Danceability vs Energy colored by genre

---

## Requirements

Primary Python libraries used:

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Recommended: create a virtual environment before installing dependencies.

---

## Installation

1. Clone the repository:

   git clone https://github.com/PrachiSukreSS/mlminiproject.git
   cd mlminiproject

2. (Optional) Create and activate a virtual environment:

   python -m venv venv
   # On Linux/macOS
   source venv/bin/activate
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

3. Install dependencies:

   pip install -r requirements.txt

If `requirements.txt` is not provided, install the packages directly:

   pip install pandas numpy matplotlib seaborn scikit-learn

---

## Running the Project

Run the main script:

   python ML_Miniproject.py

Notes:
- The script will print a sample of the generated data, the best k value and classification report.
- Several Matplotlib/Seaborn windows will open for the visualizations. If running in a headless environment (CI or server), either save the figures or run in a notebook/interactive environment.

To run in a Jupyter notebook, copy the contents of `ML_Miniproject.py` into a notebook cell and execute.

---

## Outputs and Visualizations

The script produces the following outputs:

1. Console output:
   - Sample rows of the generated dataset
   - Best k (based on test set accuracy)
   - Classification report (precision, recall, f1-score) for each genre

2. Plots:
   - Confusion matrix heatmap
   - Accuracy vs K line chart
   - Feature correlation heatmap
   - 2D scatter plot of Danceability vs Energy colored by Genre

These visualizations help to understand model performance and feature relationships.

---

## Results

Because the dataset is synthetic and random, exact results vary by run. The model selection step prints the best `k` found (from 1 to 14) and its test accuracy. The classification report and confusion matrix provide per-genre performance.

Example (sample output may vary):
- Best k value: 5 with Accuracy = 28.00%
- Classification report showing precision/recall/f1 for Pop, Rock, Classical, Jazz

---

## Limitations

- Synthetic data: Features are randomly generated and do not represent real-world audio feature distributions — results are not indicative of real music genre classification performance.
- Small dataset (200 samples): Limits model generalization.
- KNN with default distance metric (Euclidean) may not be ideal for categorical or heterogeneously scaled features — standardization is applied but some features (tempo, duration) may require better feature engineering.
- No hyperparameter tuning beyond simple grid search on `k`.
- No cross-validation; only a single train/test split is used.

