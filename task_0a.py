import sqlite3
import os
import csv
from task_1_phase1 import extract_features
from task_2_phase1 import extract_bof_hog, extract_bof_hof
from task_3_phase1 import extract_color_hist

# Path to the CSV file that contains video information
csv_file = "video_ids.csv"

# Connect to SQLite database
conn = sqlite3.connect('database1.db')
cursor = conn.cursor()

# Function to insert video features into the database
def insert_features(id, video_path, category, features):
    # Unpack features
    features_r3d, features_hog, features_hof, features_color_hist = features

    # Convert features to the appropriate format
    final_features = (
        features_r3d[0].numpy(),
        features_r3d[1].numpy(),
        features_r3d[2].numpy(),
        features_hog.tobytes(),
        features_hof.tobytes(),
        features_color_hist.tobytes()
    )

    cursor.execute('''
        INSERT INTO video_features (id, video_path, category, feature_vector_r3d_layer3, feature_vector_r3d_layer4, feature_vector_r3d_avgpool, feature_vector_hog, feature_vector_hof, feature_vector_color_hist)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (id, video_path, category, *final_features))

# Process videos based on their IDs and type (target/non-target)
def process_videos(target_type=None):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = int(row['id'])
            video_path = row['video_path']

            # Determine if the video is even, odd, or non-target
            if (target_type == 'target' and video_id % 2 == 0 and 'target_videos' in video_path) or \
               (target_type == 'target' and video_id % 2 != 0 and 'target_videos' in video_path) or \
               (target_type == 'non-target' and 'non_target_videos' in video_path):
                
                # Extract features from various visual spaces
                features_r3d = extract_features(video_path)
                txt_file_path = video_path.replace('/target_videos/', '/hmdb51_org_stips/').replace('/non_target_videos/', '/hmdb51_org_stips/').replace('.avi', '.avi.txt')
                features_hog = extract_bof_hog(video_path, txt_file_path)
                features_hof = extract_bof_hof(video_path, txt_file_path)
                features_color_hist = extract_color_hist(video_path)

                # Insert features into the database
                category = os.path.basename(os.path.dirname(video_path)) if 'target_videos' in video_path else None
                insert_features(video_id, video_path, category, (features_r3d, features_hog, features_hof, features_color_hist))

try:
    # Process even-numbered and odd-numbered target videos
    process_videos(target_type='target')
    # Process all videos in the non-target dataset
    process_videos(target_type='non-target')
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Commit changes and close the database connection after all processing
    conn.commit()
    conn.close()

print("Feature extraction and database insertion completed.")
