import argparse
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.linalg import eigh
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

class SpectralClustering:
    def __init__(self, n_clusters=2, affinity='rbf', gamma=1.0):
        """
        Initialize Spectral Clustering algorithm
        
        Parameters:
        -----------
        n_clusters : int, default=2
            Number of clusters to form
        
        affinity : str, default='rbf'
            Kernel to use for constructing similarity matrix
        
        gamma : float, default=1.0
            Kernel coefficient for rbf kernel
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
    
    def _compute_affinity_matrix(self, X):
        """
        Compute affinity matrix using RBF kernel
        
        Parameters:
        -----------
        X : numpy array
            Input feature matrix
        
        Returns:
        --------
        W : numpy array
            Affinity (similarity) matrix
        """
        # Compute pairwise squared Euclidean distances
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        
        # Compute RBF kernel
        W = np.exp(-self.gamma * sq_dists)
        
        return W
    
    def _normalize_laplacian(self, W):
        """
        Compute normalized Laplacian matrix
        
        Parameters:
        -----------
        W : numpy array
            Affinity matrix
        
        Returns:
        --------
        L : numpy array
            Normalized Laplacian matrix
        """
        # Compute degree matrix
        D = np.diag(np.sum(W, axis=1))
        
        # Compute symmetric normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        
        return L
    
    def fit(self, X):
        """
        Perform spectral clustering
        
        Parameters:
        -----------
        X : numpy array
            Input feature matrix
        
        Returns:
        --------
        labels : numpy array
            Cluster labels for each data point
        """
        # Compute affinity matrix
        W = self._compute_affinity_matrix(X)
        
        # Compute normalized Laplacian
        L = self._normalize_laplacian(W)
        
        # Compute eigenvectors of Laplacian
        eigenvalues, eigenvectors = eigh(L)
        
        # Select the first n_clusters eigenvectors corresponding to smallest eigenvalues
        # Skip the first (constant) eigenvector
        selected_eigenvectors = eigenvectors[:, 1:self.n_clusters+1]
        
        # Normalize rows to unit length
        eigenvector_rows = selected_eigenvectors / np.linalg.norm(selected_eigenvectors, axis=1)[:, np.newaxis]
        
        # Perform k-means on the eigenvector rows
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(eigenvector_rows)
        
        return labels

class VideoClusterAnalyzer:
    def __init__(self, db_path, n_clusters=3):
        """
        Initialize Video Clustering Analyzer
        
        Parameters:
        -----------
        db_path : str
            Path to the SQLite database
        n_clusters : int, default=3
            Number of clusters to form
        """
        self.db_path = db_path
        self.n_clusters = n_clusters
        self.data = None
        self.features = None
        self.labels = None
        self.clusters = None
    
    def load_data(self, label=None):
        """
        Load video database for a specific label, focusing on even-numbered videos
        
        Parameters:
        -----------
        label : str, optional
            Specific label to filter videos
        
        Returns:
        --------
        self
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        
        # Base query to select even-numbered videos
        query = """
        SELECT * FROM video_features
        WHERE id % 2 = 0
        """
        
        # Add label filter if provided
        params = []
        if label:
            query += " AND category = ?"
            params.append(label)
        
        # Read data
        self.data = pd.read_sql_query(query, conn, params=params)
        
        if self.data.empty:
            print(f"No data found for label: {label}")
        conn.close()
        return self
    
    def extract_features(self, feature_columns=None):
        """
        Extract features for clustering
        
        Parameters:
        -----------
        feature_columns : list, optional
            Columns to use for feature extraction
        
        Returns:
        --------
        self
        """
        if feature_columns is None:
            # Use numerical columns for feature extraction
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = numerical_cols
        
        # Extract and normalize features
        self.features = self.data[feature_columns].values
        
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        return self
    
    def perform_clustering(self):
        """
        Perform spectral clustering on video features
        
        Returns:
        --------
        self
        """
        # Perform spectral clustering
        spectral_clusterer = SpectralClustering(
            n_clusters=self.n_clusters, 
            affinity='rbf', 
            gamma=1.0
        )
        
        # Get cluster labels
        self.labels = spectral_clusterer.fit(self.features)
        
        # Add cluster labels to dataframe
        self.data['cluster'] = self.labels
        
        return self
    
    def visualize_clusters(self, output_path='video_clusters_mds.png'):
        """
        Visualize clusters using Multidimensional Scaling (MDS)
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save visualization
        
        Returns:
        --------
        self
        """
        # Perform MDS
        mds = MDS(n_components=2, random_state=42)
        reduced_features = mds.fit_transform(self.features)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            c=self.labels, 
            cmap='viridis',
            alpha=0.7
        )
        plt.title('Video Clusters - MDS Visualization')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.colorbar(scatter, label='Cluster')
        
        # Save plot
        plt.savefig(output_path)
        plt.close()
        
        return self
    
    def analyze_clusters(self):
        """
        Analyze and summarize clusters
        
        Returns:
        --------
        dict
            Cluster summary information
        """
        if self.labels is None:
            raise ValueError("Clustering must be performed first")
        
        # Group videos by clusters
        self.clusters = {}
        for i in range(self.n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            
            self.clusters[i] = {
                'size': len(cluster_data),
                'representative_videos': cluster_data.head(3).to_dict('records'),
                'avg_metrics': cluster_data.mean(numeric_only=True).to_dict()
            }
        
        return self.clusters

def main(db_path='database1.db'):
    parser = argparse.ArgumentParser(description="Video Cluster Analysis Tool")
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters for analysis.')
    
    args = parser.parse_args()

    n_clusters = args.n_clusters
    
    # Get unique labels
    conn = sqlite3.connect(db_path)
    labels_query = "SELECT DISTINCT category FROM video_features"
    labels = pd.read_sql_query(labels_query, conn)['category'].tolist()
    conn.close()
    
    labels = [label for label in labels if label is not None]

    # Store results for all labels
    all_label_results = {}
    
    # Perform analysis for each unique label
    for label in labels:
        print(f"\nAnalyzing label: {label}")
        
        # Initialize and run analysis
        analyzer = VideoClusterAnalyzer(db_path, n_clusters)
        analyzer.load_data(label=label)
        
        # Skip if no data for this label
        if len(analyzer.data) == 0:
            print(f"No videos found for label: {label}")
            continue
        
        (analyzer
         .extract_features()
         .perform_clustering()
         .visualize_clusters(f'video_clusters_mds_{label}.png')
        )
        
        # Analyze and print cluster details
        clusters = analyzer.analyze_clusters()
        for cluster_id, cluster_info in clusters.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {cluster_info['size']}")
            print("  Representative Videos:")
            for video in cluster_info['representative_videos']:
                print(f"    - Video ID: {video['id']}")
        
        # Store results
        all_label_results[label] = clusters
    
    return all_label_results

if __name__ == "__main__":
    main()


