# Prodigy_ML_02

# Customer Segmentation with K-means Clustering

This project implements a K-means clustering algorithm to segment customers based on their purchase history. By analyzing the data and applying K-means clustering, we can categorize retail store customers into distinct segments, gaining insights into their spending behaviors.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)


## Dataset

The dataset used for this project is the [Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) dataset from Kaggle. It contains information about customers, including their annual income and spending score.

## Installation

1. Clone this repository to your local machine:

2. Navigate to the project directory:

3. Install the required Python libraries using pip:

## Usage

1. Download the dataset from the Dataset link given below and save it as `Mall_Customers.csv` in the project directory.

2. Dataset :- https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

3. Run the Python script: python cluster_algorithm.py 
 
4. Follow the on-screen instructions to analyze the clusters and interpret the results.

## Methodology

1. Data Preprocessing: The dataset is loaded and standardized to prepare it for clustering.

2. Determining K: The Elbow Method is used to find the optimal number of clusters (K).

3. K-means Clustering: The K-means algorithm is applied to cluster customers based on their annual income and spending score.

4. Visualization: The results are visualized using Matplotlib, including scatter plots of clusters and centroids.

## Results

The script will provide insights into customer segmentation, including summary statistics for each cluster. You can adjust the `optimal_k` variable in the code to choose the desired number of clusters based on the Elbow Method graph.



