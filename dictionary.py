# %%
import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from imutils import build_montages

# %%
print("[INFO] Loading face encodings...")
data = pickle.loads(open("encodings1.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# %%
n_components_values = list(range(1, min(len(encodings[0]), len(encodings)) + 1))

explained_variances = []

for n_component in n_components_values:
    pca = PCA(n_components=n_component)
    pca.fit(encodings)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# %%

# Load data and encodings (make sure these are defined)
# For example: data = [{"imagePath": "path/to/image.jpg", "loc": (top, right, bottom, left)}, ...]
# encodings = np.array([...])

# Perform PCA to reduce dimensionality (you can adjust the number of components)
pca = PCA(n_components=12)
reduced_data = pca.fit_transform(encodings)

# Cluster the embeddings using AgglomerativeClustering
n_clusters = 30  # Set the desired number of clusters
clt = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
clt.fit(reduced_data)

# Create a DataFrame to store clustering results
df = pd.DataFrame(columns=["image path", "encoding", "initial cluster", "secondary cluster", "person", 'loc'])

# Populate the DataFrame with initial clustering results
rows = []
for i, (encoding, label) in enumerate(zip(encodings, clt.labels_)):
    row = {
        "image path": data[i]["imagePath"],
        "encoding": encoding,
        "initial cluster": label,
        "secondary cluster": None,  # Will be filled during secondary clustering
        "person": None,  # Placeholder for manual labeling
        "loc": data[i]["loc"]
    }
    rows.append(row)

df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# Save initial cluster labels and reduced data for later use (if needed)
# np.save('initial_labels.npy', clt.labels_)
# np.save('reduced_data.npy', reduced_data)

# Determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(labelIDs)
print("[INFO] Number of unique faces: {}".format(numUniqueFaces))

# Loop over the unique face integers for visualization
for labelID in labelIDs:
    print("[INFO] Faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    
    faces = []
    
    for i in idxs:
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)
    
    montage = build_montages(faces, (96, 96), (5, 5))[0]
    rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
    
    title = "Face ID #{}".format(labelID)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# %%
df

# %%

# Perform secondary clustering within each initial cluster
n_secondary_clusters = 10  # Set the desired number of secondary clusters

for initial_label in np.unique(clt.labels_):
    print(f"[INFO] Processing initial cluster {initial_label}...")

    cluster_idxs = df[df['initial cluster'] == initial_label].index
    cluster_data = np.array(df.loc[cluster_idxs, "encoding"].tolist())  # Extract encodings for the current cluster
    
    # Adjust number of secondary clusters based on the number of samples in the current cluster
    current_n_clusters = min(n_secondary_clusters, len(cluster_data))
    
    if current_n_clusters > 9:  # Ensure we have more than 1 cluster
        secondary_clt = AgglomerativeClustering(n_clusters=current_n_clusters, metric='euclidean', linkage='ward')
        secondary_labels = secondary_clt.fit_predict(cluster_data)
        
        # Update the secondary_cluster column in the DataFrame
        df.loc[cluster_idxs, "secondary cluster"] = secondary_labels
        
        # Print the number of faces in each secondary cluster
        for sec_label in np.unique(secondary_labels):
            print(f"[INFO] Number of faces in secondary cluster {sec_label} within initial cluster {initial_label}: {np.sum(secondary_labels == sec_label)}")
        
        # Visualization of secondary clusters
        plt.figure(figsize=(10, 7))
        for sec_label in np.unique(secondary_labels):
            plt.scatter(cluster_data[secondary_labels == sec_label, 0], cluster_data[secondary_labels == sec_label, 1], label=f'Secondary Cluster {sec_label}')
        
        plt.title(f'Secondary Clustering within Initial Cluster {initial_label}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        """# Create montages for each secondary cluster within the initial cluster
        for sec_label in np.unique(secondary_labels):
            print(f"[INFO] Faces for secondary cluster {sec_label} within initial cluster {initial_label}...")
            sec_cluster_idxs = cluster_idxs[secondary_labels == sec_label]
            idxs = np.random.choice(sec_cluster_idxs, size=min(25, len(sec_cluster_idxs)), replace=False)
            
            faces = []
            for i in idxs:
                image = cv2.imread(df.loc[i, "image path"])
                (top, right, bottom, left) = data[i]["loc"]
                face = image[top:bottom, left:right]
                face = cv2.resize(face, (96, 96))
                faces.append(face)
            
            montage = build_montages(faces, (96, 96), (5, 5))[0]
            rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
            
            title = f"Secondary Cluster {sec_label} in Initial Cluster {initial_label}"
            plt.imshow(rgb_image)
            plt.title(title)
            plt.axis('off')
            plt.show()"""
    else:
        print(f"[INFO] Skipping secondary clustering for initial cluster {initial_label} because it has too few samples.")

# Print the number of images per secondary cluster within each initial cluster
print("Secondary Cluster counts:")
for initial_label in np.unique(clt.labels_):
    print(f"[INFO] Secondary clusters within initial cluster {initial_label}:")
    cluster_data = np.array(df[df['initial cluster'] == initial_label]["encoding"].tolist())
    current_n_clusters = min(n_secondary_clusters, len(cluster_data))
    
    if current_n_clusters > 9:
        secondary_clt = AgglomerativeClustering(n_clusters=current_n_clusters, metric='euclidean', linkage='ward')
        secondary_labels = secondary_clt.fit_predict(cluster_data)
        
        cluster_counts = {label: 0 for label in np.unique(secondary_labels)}
        for sec_label in np.unique(secondary_labels):
            count = np.sum(secondary_labels == sec_label)
            cluster_counts[sec_label] = count
            print(f"Secondary Cluster {sec_label}: {count} faces")

# %%

# Save the DataFrame to a CSV file for later use
#df.to_csv("clustering_results.csv", index=False)

print("[INFO] Clustering results saved to 'clustering_results.csv'.")


# %%
df

# %%

# Define the initial cluster you want to visualize
initial_cluster_label = 4

# Filter the DataFrame for the specified initial cluster
filtered_df = df[df['initial cluster'] == initial_cluster_label]

# Get the unique secondary clusters within the initial cluster
secondary_cluster_labels = filtered_df['secondary cluster'].unique()

# Iterate over each secondary cluster and create a montage
for secondary_cluster_label in secondary_cluster_labels:
    # Filter the DataFrame for the current secondary cluster
    sec_cluster_df = filtered_df[filtered_df['secondary cluster'] == secondary_cluster_label]
    
    # Randomly sample up to 25 images from the filtered DataFrame
    sample_size = min(25, len(sec_cluster_df))
    sampled_df = sec_cluster_df.sample(n=sample_size, random_state=42)
    
    faces = []

    # Loop over the sampled DataFrame rows to extract and process the face images
    for _, row in sampled_df.iterrows():
        image_path = row["image path"]
        image = cv2.imread(image_path)
        # Assuming the 'loc' column is in your DataFrame and contains the face bounding box
        (top, right, bottom, left) = row["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # Create a montage using 96x96 "tiles" with 5 rows and 5 columns
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        
        title = f"Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}"
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')  # Optional: Turn off the axis labels
        plt.show()
    else:
        print(f"[INFO] No images found for Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}")

# %%

# Define the initial cluster you want to visualize
initial_cluster_label = 1

# Filter the DataFrame for the specified initial cluster
filtered_df = df[df['initial cluster'] == initial_cluster_label]

# Get the unique secondary clusters within the initial cluster
secondary_cluster_labels = filtered_df['secondary cluster'].unique()

# Iterate over each secondary cluster and create a montage
for secondary_cluster_label in secondary_cluster_labels:
    # Filter the DataFrame for the current secondary cluster
    sec_cluster_df = filtered_df[filtered_df['secondary cluster'] == secondary_cluster_label]
    
    # Randomly sample up to 25 images from the filtered DataFrame
    sample_size = min(25, len(sec_cluster_df))
    sampled_df = sec_cluster_df.sample(n=sample_size, random_state=42)
    
    faces = []

    # Loop over the sampled DataFrame rows to extract and process the face images
    for _, row in sampled_df.iterrows():
        image_path = row["image path"]
        image = cv2.imread(image_path)
        # Assuming the 'loc' column is in your DataFrame and contains the face bounding box
        (top, right, bottom, left) = row["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # Create a montage using 96x96 "tiles" with 5 rows and 5 columns
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        
        title = f"Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}"
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')  # Optional: Turn off the axis labels
        plt.show()
    else:
        print(f"[INFO] No images found for Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}")

# %%

# Define the initial cluster you want to visualize
initial_cluster_label = 2

# Filter the DataFrame for the specified initial cluster
filtered_df = df[df['initial cluster'] == initial_cluster_label]

# Get the unique secondary clusters within the initial cluster
secondary_cluster_labels = filtered_df['secondary cluster'].unique()

# Iterate over each secondary cluster and create a montage
for secondary_cluster_label in secondary_cluster_labels:
    # Filter the DataFrame for the current secondary cluster
    sec_cluster_df = filtered_df[filtered_df['secondary cluster'] == secondary_cluster_label]
    
    # Randomly sample up to 25 images from the filtered DataFrame
    sample_size = min(25, len(sec_cluster_df))
    sampled_df = sec_cluster_df.sample(n=sample_size, random_state=42)
    
    faces = []

    # Loop over the sampled DataFrame rows to extract and process the face images
    for _, row in sampled_df.iterrows():
        image_path = row["image path"]
        image = cv2.imread(image_path)
        # Assuming the 'loc' column is in your DataFrame and contains the face bounding box
        (top, right, bottom, left) = row["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # Create a montage using 96x96 "tiles" with 5 rows and 5 columns
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        
        title = f"Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}"
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')  # Optional: Turn off the axis labels
        plt.show()
    else:
        print(f"[INFO] No images found for Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}")

# %%

# Define the initial cluster you want to visualize
initial_cluster_label = 3

# Filter the DataFrame for the specified initial cluster
filtered_df = df[df['initial cluster'] == initial_cluster_label]

# Get the unique secondary clusters within the initial cluster
secondary_cluster_labels = filtered_df['secondary cluster'].unique()

# Iterate over each secondary cluster and create a montage
for secondary_cluster_label in secondary_cluster_labels:
    # Filter the DataFrame for the current secondary cluster
    sec_cluster_df = filtered_df[filtered_df['secondary cluster'] == secondary_cluster_label]
    
    # Randomly sample up to 25 images from the filtered DataFrame
    sample_size = min(25, len(sec_cluster_df))
    sampled_df = sec_cluster_df.sample(n=sample_size, random_state=42)
    
    faces = []

    # Loop over the sampled DataFrame rows to extract and process the face images
    for _, row in sampled_df.iterrows():
        image_path = row["image path"]
        image = cv2.imread(image_path)
        # Assuming the 'loc' column is in your DataFrame and contains the face bounding box
        (top, right, bottom, left) = row["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # Create a montage using 96x96 "tiles" with 5 rows and 5 columns
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        
        title = f"Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}"
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')  # Optional: Turn off the axis labels
        plt.show()
    else:
        print(f"[INFO] No images found for Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}")

# %%

# Define the initial cluster you want to visualize
initial_cluster_label = 4

# Filter the DataFrame for the specified initial cluster
filtered_df = df[df['initial cluster'] == initial_cluster_label]

# Get the unique secondary clusters within the initial cluster
secondary_cluster_labels = filtered_df['secondary cluster'].unique()

# Iterate over each secondary cluster and create a montage
for secondary_cluster_label in secondary_cluster_labels:
    # Filter the DataFrame for the current secondary cluster
    sec_cluster_df = filtered_df[filtered_df['secondary cluster'] == secondary_cluster_label]
    
    # Randomly sample up to 25 images from the filtered DataFrame
    sample_size = min(25, len(sec_cluster_df))
    sampled_df = sec_cluster_df.sample(n=sample_size, random_state=42)
    
    faces = []

    # Loop over the sampled DataFrame rows to extract and process the face images
    for _, row in sampled_df.iterrows():
        image_path = row["image path"]
        image = cv2.imread(image_path)
        # Assuming the 'loc' column is in your DataFrame and contains the face bounding box
        (top, right, bottom, left) = row["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    # Create a montage using 96x96 "tiles" with 5 rows and 5 columns
    if faces:
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        rgb_image = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
        
        title = f"Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}"
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')  # Optional: Turn off the axis labels
        plt.show()
    else:
        print(f"[INFO] No images found for Secondary Cluster {secondary_cluster_label} in Initial Cluster {initial_cluster_label}")


