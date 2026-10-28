# sar-oil-spill-eda
Exploratory Data Analysis (EDA) of the Sentinel-1 SAR Oil Spill dataset, analyzing visual patterns and pixel statistics to distinguish oil spills from normal sea surfaces.
.
.
# Sentinel-1 SAR Oil Spill Detection - EDA & Baseline CNN

## Project Overview
This repository contains an Exploratory Data Analysis (EDA) and a baseline Convolutional Neural Network (CNN) trained on the Sentinel-1 SAR Oil Spill Detection dataset. The project analyzes visual patterns, pixel intensity statistics, and trains a model to classify ocean surface radar backscatter as either an oil spill or a non-oil spill.

## Dataset
*   **Source:** [Kaggle - Sentinel-1 SAR Oil Spill Dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset)
*   **Characteristics:** Grayscale Synthetic Aperture Radar (SAR) imagery. Oil spills appear as dark patches due to reduced radar backscatter from dampened capillary waves.

## Repository Structure
*   `src/` - Contains the Python scripts for EDA and Model Training.
*   `notebooks/` - Contains the interactive Jupyter Notebook (if applicable).
*   `data/` - Git-ignored directory for raw dataset images.
*   `docs/` - Project documentation and report resources.

## How to Run
1. Clone the repository and navigate to the project folder.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install tensorflow matplotlib opencv-python pandas numpy`
4. Download the Kaggle dataset and place the class folders inside the `data/` directory.
5. Run the training script: `python src/train_model.py`

## Full Report
Detailed EDA findings, statistical outliers, and model performance interpretations can be found in the(.https://docs.google.com/document/d/1pqIVngSQ6B8h9rZ_Mwt6EJSphsv-ngbXQYSD9PibVEc/edit?usp=sharing).
