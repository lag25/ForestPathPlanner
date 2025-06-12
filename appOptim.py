import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import pathfinder # My C++ based path finding module
from threshold import IsoGrayThresh

def main():
    # --- Configuration ---
    IMAGE_PATH = 'img3.jpeg'  # <--- CHANGE THIS TO WHATEVER IMAGE YOU WANT I SWEAR I WILL ADD A NEATER METHOD FOR INPUT LATER 
    START_PIXEL = (0, 0) # <--- WHATEVER START PIXEL YOU WANT AS LONG AS ITS WITHIN THE DIMENSIONS OF THE IMAGE 
    
    # --- Load Image ---
    try:
        # Open the image and convert to grayscale
        img = plt.imread(IMAGE_PATH)
        img = IsoGrayThresh(img)
        img = Image.fromarray(img).convert('L')
        img_array = np.array(img)
        
        # Define target based on image dimensions
        rows, cols = img_array.shape
        target_pixel = (rows - 1, cols - 1)

    except FileNotFoundError:
        print(f"Error: Image not found at {IMAGE_PATH}")
        # Create a dummy image for demonstration if not found
        print("Creating a dummy gradient image for demonstration.")
        rows, cols = 400, 600
        target_pixel = (rows - 1, cols - 1)
        x = np.linspace(0, 255, cols)
        y = np.linspace(0, 255, rows)
        xv, yv = np.meshgrid(x, y)
        img_array = (xv + yv).astype(np.uint8)
        img = Image.fromarray(img_array)


    # ---
    # THE HEAVY LIFTING IS DONE HERE!
    # ---
    # 1. Create an instance of our C++ class
    path_calculator = pathfinder.OptimalPathing()

    # 2. Call the C++ function to compute the path
    time1 = time.time()
    print("Computing path using C++ module...")
    # The C++ function takes the numpy array and the start/target tuples
    shortest_path = path_calculator.compute_path(img_array, START_PIXEL, target_pixel)
    print("Path computation complete.")
    print(f"in {time.time()-time1} seconds")
    if not shortest_path:
        print("No path was found.")
        return

    # --- Visualization ---
    print(f"Path found with {len(shortest_path)} points. Visualizing...")
    x_coords, y_coords = zip(*shortest_path)  # Extract x, y coordinates

    plt.figure(figsize=(15, 10))
    plt.imshow(img, cmap='gray')
    # Note: plt.plot uses (x, y) which corresponds to (col, row)
    plt.plot(y_coords, x_coords, color='cyan', linewidth=2.5, label='Optimal Path')
    
    # Mark start and end points
    plt.scatter([START_PIXEL[1]], [START_PIXEL[0]], color='lime', s=100, zorder=5, label='Start')
    plt.scatter([target_pixel[1]], [target_pixel[0]], color='red', s=100, zorder=5, label='Target')

    plt.title('Optimal Path found by C++ Module')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
