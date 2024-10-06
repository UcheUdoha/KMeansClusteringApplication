import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the dataset: handle missing values, select relevant columns, and normalize data.
    """
    # Select relevant columns (example: selecting numerical columns for clustering)
    numerical_cols = ['LAT', 'LON', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']
    df = df[numerical_cols]

    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Normalize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled


def perform_kmeans(data, n_clusters):
    """
    Perform K-Means clustering on the dataset.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans


def plot_clusters(data, kmeans):
    """
    Plot the clustered data.
    """
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('K-Means Clustering')
    plt.show()


def main(file_path, n_clusters):
    df = load_data(file_path)
    data = preprocess_data(df)
    kmeans = perform_kmeans(data, n_clusters)
    plot_clusters(data, kmeans)


if __name__ == "__main__":
    file_path = 'patients.csv'  # Replace with your dataset path
    n_clusters = 3  # Example number of clusters
    main(file_path, n_clusters)
