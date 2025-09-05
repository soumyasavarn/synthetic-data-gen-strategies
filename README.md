# Synthetic Data Generation Strategies

This project was developed as part of **DA312: Advanced Machine Learning Laboratory (Jan–Apr 2025)**.
It focuses on building and evaluating GAN-based models for generating synthetic data in both **tabular** and **image domains**.

---

## Features

* **CTGAN for Tabular Data**

  * Supports custom CSV files.
  * Provides options to select between Vanilla GAN and CTGAN.
  * Allows configuration of training parameters such as number of epochs and number of synthetic samples.
  * Models are trained on the fly; no pre-trained weights are used.

* **Pix2Pix for Image-to-Comic Translation**

  * Uses a pre-trained model `generator_120.pth`.
  * Model was trained on the [Comic Faces Paired Synthetic v2 dataset](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2) using a Tesla P100 GPU for approximately 10 hours (120 epochs).
  * Accepts `.jpg` and `.png` image formats as input.

* **Data Augmentation on the Iris Dataset**

  * Applied augmentation techniques to improve classification accuracy.
  * Achieved a **3.33% improvement in accuracy** compared to the baseline.

* **Streamlit Application**

  * Provides a single frontend interface (`app.py`) for both image and tabular data synthesis.
  * Automatically detects the type of uploaded file (CSV or image) and runs the appropriate pipeline.
  * For tabular data: trains a GAN model with user-specified parameters.
  * For images: uses the pre-trained Pix2Pix generator for comic-style image translation.

---

## Project Structure

* `app.py`: Streamlit frontend.
* `testing_data/`: Example CSV and image files for quick testing.
* `generator_120.pth`: Pre-trained Pix2Pix generator (image-to-comic model).

---

## Running the Application

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Upload a `.csv` file for tabular data generation or a `.jpg`/`.png` file for image translation.

**Note:** For best results with image generation, use the provided sample data in `testing_data/`. GAN models trained in this setup may not generalize well and are prone to overfitting to their training domain.

---

## Report

The detailed report can be found here:
[Project Report (Google Drive)](https://drive.google.com/file/d/1Y6Hx2FJsqoxIJaOXLrbAghCdvE5GtLds/view)
