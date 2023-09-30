import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from sklearn.cluster import KMeans

def get_rgb(image_dir):
    # Open the image
    img = Image.open(image_dir)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Get dimensions
    width, height = img.size

    # Initialize a list to hold all the rows
    all_rows = []

    # Loop through the image
    for ih in range(height):
        row_data = []
        for iw in range(width):
            rgb = img.getpixel((iw, ih))
            row_data.append(rgb)
        all_rows.append(row_data)

    # Setup Dataframe
    df = pd.DataFrame(all_rows)
    df.columns = list(range(width))
    df.index.name = 'Height / Width'
    return df

def average_color(file_dir):
    # Step 1: Extract images from the Excel file
    excel_file = file_dir
    image_files = []

    with ZipFile(excel_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith('xl/media/'):
                with zip_ref.open(file_info.filename) as img_file:
                    img_bytes = img_file.read()
                    image_files.append(img_bytes)

    # Step 2: Read names using pandas
    df_names = pd.read_excel(excel_file, usecols=["Name"])  # Assuming 'Name' column exists
    names = df_names["Name"].tolist()

    # Step 3: Calculate average color for each image
    avg_colors = []

    for img_bytes in image_files:
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        np_img = np.array(img)
        avg_rgb = np_img.mean(axis=(0, 1))
        avg_colors.append(tuple(map(int, avg_rgb)))

    # Create DataFrame with mismatched lengths
    df_avg_colors = pd.DataFrame({
        'Name': names,
        'AvgColor': avg_colors + [None] * (len(names) - len(avg_colors))
    })

    # Detect names that don't have corresponding images
    missing_images = df_avg_colors[df_avg_colors['AvgColor'].isna()]
    print('WARNING!')
    print(f'There are some missing images: {missing_images}')

    return df_avg_colors.dropna()

def get_dominant_color(image, k=1, ignore_color=(139, 139, 139)):
    pixels = np.array(image).reshape(-1, 3)
    pixels = pixels[np.any(pixels != ignore_color, axis=1)]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = kmeans.cluster_centers_[labels[np.argmax(counts)]].astype(int)
    return tuple(dominant_color)


def dominant_color(file_dir):
    # Step 1: Extract images from the Excel file
    image_files = []
    with ZipFile(file_dir, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith('xl/media/'):
                with zip_ref.open(file_info.filename) as img_file:
                    img_bytes = img_file.read()
                    image_files.append(img_bytes)

    # Step 2: Read names using pandas
    df_names = pd.read_excel(file_dir, usecols=["Name"])
    names = df_names["Name"].tolist()

    # Step 3: Calculate dominant color for each image
    dominant_colors = []
    for img_bytes in image_files:
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        dominant_color = get_dominant_color(img)
        dominant_colors.append(dominant_color)

    # Step 4: Create DataFrame
    df_dominant_colors = pd.DataFrame({
        'Name': names,
        'DominantColor': dominant_colors + [None] * (len(names) - len(dominant_colors))
    })

    # Detect names that don't have corresponding images
    missing_images = df_dominant_colors[df_dominant_colors['DominantColor'].isna()]
    print('WARNING!')
    print(f'There are some missing images: {missing_images}')

    return df_dominant_colors.dropna()