import argparse
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

class LabelClusterAnalyzer:
    def __init__(self, db_path, n_clusters):
        self.db_path = db_path
        self.n_clusters = n_clusters
        self.labels = None
        self.label_vectors = None
        self.labels_clustered = None
        self.clusters = None
    
    def load_labels(self):
        """
        Load unique labels from the database
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        
        # Get unique labels
        labels_query = "SELECT DISTINCT category FROM video_features"
        self.labels = pd.read_sql_query(labels_query, conn)['category'].tolist()
        
        # Filter out None values
        self.labels = [label for label in self.labels if label is not None]
        
        conn.close()
        return self
    
    def extract_label_features(self):
        """
        Convert labels to numerical vectors using CountVectorizer
        """
        vectorizer = CountVectorizer()
        self.label_vectors = vectorizer.fit_transform(self.labels).toarray()
        return self
    
    def perform_clustering(self):
        """
        Perform k-means clustering on label vectors
        """
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_clustered = kmeans.fit_predict(self.label_vectors)
        return self
    
    def visualize_clusters(self, output_path='label_clusters_mds.png'):
        """
        Visualize clusters using Multidimensional Scaling (MDS)
        """
        # Perform MDS
        mds = MDS(n_components=2, random_state=42)
        reduced_vectors = mds.fit_transform(self.label_vectors)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_vectors[:, 0], 
            reduced_vectors[:, 1], 
            c=self.labels_clustered, 
            cmap='viridis',
            alpha=0.7
        )
        plt.title('Label Clusters - MDS Visualization')
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
        """
        if self.labels_clustered is None:
            raise ValueError("Clustering must be performed first")
        
        # Group labels by clusters
        self.clusters = {}
        for i in range(self.n_clusters):
            cluster_labels = [label for label, cluster in zip(self.labels, self.labels_clustered) if cluster == i]
            
            self.clusters[i] = {
                'size': len(cluster_labels),
                'labels': cluster_labels
            }
        
        return self.clusters

def main(n_clusters=3):
    db_path = 'database1.db'
    
    # Initialize and run analysis
    analyzer = LabelClusterAnalyzer(db_path, n_clusters)
    analyzer.load_labels()
    analyzer.extract_label_features()
    analyzer.perform_clustering()
    analyzer.visualize_clusters('label_clusters_mds.png')
    
    # Analyze and print cluster details
    clusters = analyzer.analyze_clusters()
    for cluster_id, cluster_info in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_info['size']}")
        print("  Labels:")
        for label in cluster_info['labels']:
            print(f"    - {label}")
    
    return clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Clustering Analysis Tool")
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters for analysis.')
    
    args = parser.parse_args()
    
    main(n_clusters=args.n_clusters)