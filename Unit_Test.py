import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from K_means import load_data, preprocess_data, perform_kmeans


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = {
            'LAT': [42.39226128, 42.26323275, 41.66503003],
            'LON': [-72.47915234, -71.48126062, -70.91450698],
            'HEALTHCARE_EXPENSES': [1318721.22, 9089.75, 230547.34],
            'HEALTHCARE_COVERAGE': [4100.44, 0, 5821.16]
        }
        self.df = pd.DataFrame(self.data)

    def test_load_data(self):
        df = load_data('patients.csv')  # Replace with a valid path
        self.assertIsInstance(df, pd.DataFrame)

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.df)
        self.assertIsInstance(processed_data, np.ndarray)
        self.assertEqual(processed_data.shape, (3, 4))  # Ensure the shape matches the sample data

    def test_perform_kmeans(self):
        processed_data = preprocess_data(self.df)
        kmeans = perform_kmeans(processed_data, 2)
        self.assertIsInstance(kmeans, KMeans)
        self.assertEqual(len(kmeans.labels_), 3)  # Ensure the number of labels matches the sample data

if __name__ == '__main__':
    unittest.main()