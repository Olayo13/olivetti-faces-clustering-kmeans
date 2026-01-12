# Olivetti Face Clustering: Unsupervised Learning with K-Means and PCA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Description

This project explores **unsupervised machine learning** techniques to uncover patterns and group facial images without using predefined labels. Using the *Olivetti Faces* dataset, the main objective was to segment images into visually coherent clusters based on facial structure, pose, and lighting conditions.

A complete pipeline is implemented to compare the performance of **K-Means** on raw pixel data versus an optimized version using **dimensionality reduction (PCA)**, demonstrating how noise removal improves cluster purity.

## Dataset

The **Olivetti Faces** dataset (available in Scikit-Learn) was used, consisting of:

* **400 grayscale images**
* **Resolution:** 64×64 pixels
* **Subjects:** 40 distinct individuals (10 images per person)
* **Variability:** Images include different facial expressions (eyes open/closed, smiling/neutral) and accessories (glasses)

## Methodology

The workflow follows a rigorous data engineering approach:

1. **Stratified Data Splitting:**
   `StratifiedShuffleSplit` was used to ensure that the Training, Validation, and Test sets contained a balanced representation of all 40 subjects.
2. **Hyperparameter Selection:**
   Determination of the optimal number of clusters ((k)) using:

   * **Elbow Method:** Inertia analysis
   * **Silhouette Score:** To measure cluster cohesion and separation
   * *Result:* A value of (k \approx 120) was selected to capture pose variations per individual.
3. **Baseline Modeling (Raw Pixels):**
   Initial training using the original 4,096 pixel features.
4. **PCA Optimization:**
   Application of **Principal Component Analysis** to retain 99% of the variance, reducing dimensionality from 4,096 to approximately 220 components and filtering high-frequency noise.
5. **3D Visualization:**
   Projection of the feature space using the first three principal components to inspect cluster distribution.

## Key Results

### 1. The Noise Problem

The model trained on raw pixels was sensitive to background noise and lighting variations, producing mixed clusters where different individuals were grouped together solely due to similar illumination.

### 2. Improvement with PCA

After applying PCA as a preprocessing step:

* **Noise Reduction:** Generated centroids became visually smoother and more defined.
* **Cluster Purity:** Identity separation improved significantly. For example, in the analysis of *Subject 22*, the optimized model grouped 4 of their images into a dominant cluster with minimal intrusion, while separating other images into different clusters based on specific gestures (smiling vs. neutral).

### 3. Latent Space Visualization

The 3D plots revealed a structure composed of a **dense core** (where average faces cluster) and **outer arms** containing faces with distinctive features (glasses, facial hair, extreme poses).

## Technologies Used

* **Language:** Python 3
* **Main Libraries:**

  * `scikit-learn`: K-Means, PCA, and validation metrics
  * `matplotlib`: Static visualization of faces and evaluation curves
  * `plotly`: Interactive 3D visualizations
  * `numpy` / `pandas`: Numerical computation and data manipulation

## How to Run

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn plotly
   ```
3. Open the main notebook:

   ```bash
   jupyter notebook Notas_P3.1.ipynb
   ```
