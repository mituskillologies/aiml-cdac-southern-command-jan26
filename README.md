# AI & Machine Learning Training - CDAC & Southern Command (Jan 2026)

This repository contains the source code, Jupyter notebooks, and datasets used during the AI & Machine Learning training program conducted for **CDAC & Southern Command** in January 2026. 

**Organized by:** [MITU Skillologies](https://mitu.co.in)  
**Instructor:** Tushar B. Kute

## 📂 Repository Structure

The course material is divided into logical modules progressing from Python basics to Advanced Deep Learning and Computer Vision.

### Module 1: Python Basics & Data Analysis
Foundational concepts of Python programming and data manipulation using Pandas and visualization tools.
- `01 Python Introduction.ipynb` - Syntax, variables, and data types.
- `02 Series and DataFrame.ipynb` - Introduction to Pandas data structures.
- `03 Data Analysis using Pandas.ipynb` - Filtering, sorting, and analyzing datasets.
- `04 Data Cleaning.ipynb` - Handling missing values and data sanitization.
- `05 EDA with matplotlib.ipynb` - Basic plotting and data visualization.
- `06 Data Visualization using Seaborn.ipynb` - Statistical data visualization.
- `07 Data Preprocessing.ipynb` - Encoding, scaling, and preparing data for ML.

### Module 2: Classical Machine Learning (Supervised & Unsupervised)
Implementation of core ML algorithms for regression, classification, and clustering.
- **Regression:**
  - `08 Simple Linear Regression.ipynb`
  - `09 Linear Regression.ipynb`
  - `10 Multiple Regression.ipynb`
  - `14 Decision Tree Regression.ipynb`
- **Classification:**
  - `11 Sigmoid Function.ipynb`
  - `12 Logistic Regression.ipynb`
  - `13 Decision Tree Classification.ipynb`
  - `15 Simple KNN.ipynb`
  - `16 K-Nearest Neighbor Classification.ipynb`
  - `17 Support Vector Machine.ipynb`
- **Clustering:**
  - `18 Unsupervised Learning - Clustering.ipynb` (K-Means)

### Module 3: Deep Learning & Neural Networks (ANN)
Introduction to Deep Learning using Keras and TensorFlow.
- `19 Activation Functions.ipynb` - Sigmoid, Tanh, ReLU, etc.
- `20 Artificial Neural Network.ipynb` - Building the first ANN.
- `21 ANN - Multiclass Classification.ipynb` - Handling multiple categories.
- `22 Regression using ANN.ipynb` - Predicting continuous values with Neural Networks.

### Module 4: Computer Vision (CNN)
Specialized architectures for image processing and object recognition.
- `23 Basic Image Processing.ipynb` - OpenCV and PIL basics.
- `24 Image Classification using ANN.ipynb` - Limitations of ANN on images.
- `25 Convolutional Neural Network.ipynb` - Building CNNs (Conv2D, MaxPooling).
- `26 CNN for Color Image Classification.ipynb` - Working with CIFAR-10.
- `27 CNN for User Data.ipynb` - Custom dataset implementation.
- `28 VGGNet.ipynb` - Using pre-trained VGG16.
- `29 ResNet50.ipynb` - Using pre-trained ResNet50.
- `30 Transfer Learning.ipynb` - Fine-tuning models for custom tasks.

### Module 5: Sequence Models (RNN & LSTM)
Handling time-series data and sequential information.
- `31 RNN and LSTM.ipynb` - Time Series Forecasting (Weather prediction).
- `32 RNN Sequence Modeling.ipynb` - Sequence prediction (Text/Character generation).

## 🛠️ Prerequisites & Installation

To run these notebooks locally, ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras opencv-python pillow
