# K-Means Clustering Application

## Overview
This project is a Python application that uses the **K-Means clustering algorithm** to explore patterns in a dataset acquired from the **Registry of Open Data on AWS**. The application aims to group data into clusters, offering valuable insights based on similarities within the dataset. 

## Features:
- **Data Import:** The dataset was selected from the AWS Registry of Open Data and imported into the Python application.
- **K-Means Clustering:** Implementation of K-Means clustering algorithm to identify natural groupings in the data.
- **Unit Testing:** Robust unit tests ensure the functionality and correctness of the clustering algorithm.
- **Executive Summary:** A one-page summary provides insights into the findings, challenges, and conclusions.

## Dataset:
- Dataset acquired from the **Registry of Open Data on AWS**.
  
## Technologies Used:
- **Python** for K-Means implementation.
- **PyCharm** as the IDE.
- **Sklearn** for clustering and data handling.
- **Pandas** for data manipulation.
- **Matplotlib/Seaborn** for visualizing clusters.

## How to Run:
1. Clone this repository.
2. Install required libraries: `pip install -r requirements.txt`
3. Run the `kmeans_clustering.py` script.
4. Unit tests can be executed via `test_kmeans.py` to validate the application.

## Results:
The application clusters the data effectively, revealing meaningful insights based on the dataset’s inherent patterns. The results are presented visually using scatter plots, with clusters differentiated by colour.

## Future Improvements:
- Experiment with different numbers of clusters.
- Apply elbow method to optimize the number of clusters.
- Improve preprocessing for better data quality and cleaner clusters.

