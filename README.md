# Modeling and Predicting Crop Health Using Remote Sensing Data and Neural Networks

## Author: Azad Rasul  
**Contact:** azad.rasul@soran.edu.iq

## Introduction
This project focuses on using remote sensing data to analyze and assess vegetation health. By leveraging raster and vector data, the project computes vegetation indices like NDVI (Normalized Difference Vegetation Index) and classifies plots based on their vegetation health. The classification model is implemented using a neural network built with TensorFlow, and various metrics are computed to evaluate its performance.

## Key Features
- **Data Handling:** Loads and processes raster data, including DEMs (Digital Elevation Models), orthophotos, and DTMs (Digital Terrain Models).
- **Vegetation Indices:** Computes NDVI to assess vegetation health.
- **Data Masking:** Masks invalid data for elevation and thermal values.
- **Zonal Statistics:** Computes mean NDVI, thermal, elevation, and DTM values for each plot.
- **Neural Network Classification:** Implements a neural network model using TensorFlow to classify plots based on vegetation health.
- **Performance Evaluation:** Evaluates the model's performance using accuracy, precision, recall, F1 score, and ROC-AUC score.

## Data Download
The dataset used in this project can be downloaded from the DroneMapper Crop Analysis Data. Extract the data into the `data/` directory in your working environment.

## Installation and Setup
Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
## Usage

### Load and Preprocess Data:
- Load DEM, orthophoto, and DTM data using `rasterio`.
- Mask invalid elevation and thermal values.
- Compute NDVI for vegetation health assessment.

### Compute Zonal Statistics:
- Calculate mean NDVI, thermal, elevation, and DTM values for each plot using the `compute_zonal_stats()` function.

### Prepare Data for Model Training:
- Create a feature matrix and a synthetic target variable focusing on healthy crops.
- Handle data imbalance by undersampling the majority class.

### Train the Neural Network Model:
- Split the data into training and testing sets.
- Standardize the features.
- Define and train the neural network model using TensorFlow.

### Evaluate Model Performance:
- Use accuracy, precision, recall, F1 score, and ROC-AUC score to assess the model's performance.

## Example
Below is an example of how to load data and train the model:
```bash
import rasterio
import geopandas as gpd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and preprocess raster data
dem = rasterio.open('data/dem.tif')
ortho = rasterio.open('data/ortho.tif')
dtm = rasterio.open('data/dtm.tif')

# Compute NDVI
ndvi = (ortho.read(4) - ortho.read(1)) / (ortho.read(4) + ortho.read(1))

# Calculate zonal statistics
plots_1 = gpd.read_file('data/plots_1.shp')
plots_1['NDVI_mean'] = compute_zonal_stats(plots_1, ndvi, dem.transform)['mean']

# Train the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(4,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)
```
## Results
The model is trained for 100 epochs, with validation accuracy consistently improving across epochs. The final model is able to classify vegetation health with high accuracy.

## Future Work
Experiment with different vegetation indices and additional remote sensing data.
Optimize the neural network architecture for better performance.
Apply the model to different crop types and geographical regions.
## License
This project is licensed under the MIT License.
