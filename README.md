# Meriti Assessment – Road Segmentation

This project implements a  U-Net model for road segmentation on the Massachusetts Roads Dataset, and experiments with Teacher-Student approach.

##  Project Structure

- `baseline_model.py` – U-Net baseline model.
- `student_training.py` – Training script for the student model with class weighting.
- `generate-pseudo-masks.py` – Script to create partial masks using background undersampling.
- `plot_detailed_results.py` – Script to generate detailed performance plots and analysis.
- `config.py` – Centralized dataset paths, download dataset from kaggle, preprocessing logic, and configuration.

## Installation

Create a virtual environment and install the dependencies:

```bash
pip install ir requirements.txt 
```

Then, to run the whole project, first run `config.py`, then the `baseline_model.py`. after that you can use `plot_detailed_results.py` to evaluate baseline model.
Next, with baseline model trained, the `generate-pseudo-mask.py`  need to excecute before `student_training.py`.
