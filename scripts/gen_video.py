import cv2
import os
import sys

def create_video_from_directory(directory, output_filename, fps=30, duration_per_image=0.5):
    # Collect all image files in the directory with common extensions (case-insensitive)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    if not image_paths:
        print(f"No images found in {directory}, skipping.")
        return
    
    image_paths.sort()  # Sort alphabetically
    
    # Read the first image to determine frame size
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error reading first image in {directory}, skipping.")
        return
    height, width, _ = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your system
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error opening video writer for {output_filename}")
        return
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path}")
            continue
        
        # Handle grayscale or alpha channel images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize to the first image's dimensions
        img_resized = cv2.resize(img, (width, height))
        
        # Calculate the number of frames to add for 0.5 seconds
        num_frames = int(duration_per_image * fps)
        
        # Write the frame multiple times
        for _ in range(num_frames):
            video_writer.write(img_resized)
    
    video_writer.release()
    print(f"Video saved to {output_filename}")

def main():
    # Get directories from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py directory1 [directory2 ...]")
        sys.exit(1)
    
    directories = sys.argv[1:]
    
    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Directory {directory} does not exist, skipping.")
            continue
        
        # Generate output filename based on directory name
        dir_name = os.path.basename(os.path.normpath(directory))
        output_filename = f"{dir_name}_output.mp4"
        
        create_video_from_directory(directory, output_filename)

if __name__ == "__main__":
    main()