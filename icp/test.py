import torch
from datetime import datetime
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else "cpu"
    # import argument parser
    import argparse
    import os
    import numpy as np
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint file")
    parser.add_argument("image_folder", type=str, help="Path to the image folder")
    parser.add_argument("--conf_threshold", type=float, default=0.45, help="Confidence threshold")

    args = parser.parse_args()

    # Load the model checkpoint     
    torch.cuda.empty_cache()
    model = YOLO(args.checkpoint)

    # make sure to get only images jpg or png
    images = [img for img in os.listdir(args.image_folder) if img.endswith(".jpg") or img.endswith(".png")] 
    # Sort images to process them in a consistent order
    images.sort()

    run_name = f"predict_yolorun_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # List to store detection counts for each frame
    detection_counts = []
    
    # Process each image individually
    for img_counter, image_name in enumerate(images):
        image_path = os.path.join(args.image_folder, image_name)
        # if the image is not square pad in the shorter dimension
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        if h != w:
            dim_diff = abs(h - w)
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h < w else ((0, 0), (pad1, pad2), (0, 0))
            image = np.pad(image, pad, mode='constant', constant_values=128)
        
        # Process single image
        with torch.no_grad():
            results = model.predict(source=image,
                                  conf=args.conf_threshold,
                                  imgsz=max(h, w),
                                  project=None,
                                  name=run_name,
                                  exist_ok=True,
                                  save=True,
                                  save_txt=True,
                                  show_boxes=True,
                                  show_labels=False,
                                  show_conf=True,
                                  device=device)
            
        # Process the result (there will be only one since we're processing one image)
        result = results[0]
        
        # Count detections in this frame
        detection_count = 0
        if hasattr(result, 'obb') and result.obb is not None:
            detection_count = result.obb.shape[0]
        detection_counts.append(detection_count)
        
        # Add detection count text to the image
        annotated_img = result.plot()
        cv2.putText(annotated_img, f"Detections: {detection_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save the images in the run_name folder with the format image{int}.jpg
        cv2.imwrite(f"runs/obb/{run_name}/image{img_counter:06d}.jpg", annotated_img)
        
        # Print progress
        if img_counter % 10 == 0:
            print(f"Processed {img_counter}/{len(images)} images")
    
    # Generate histogram of detection counts
    plt.figure(figsize=(10, 6))
    plt.hist(detection_counts, bins=max(10, max(detection_counts) + 1) if detection_counts else 10, color='blue', alpha=0.7)
    plt.title('Histogram of Whale Detections per Frame')
    plt.xlabel('Number of Detections')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save histogram
    histogram_path = f"runs/obb/{run_name}/detection_histogram.png"
    plt.savefig(histogram_path)
    print(f"Histogram saved to {histogram_path}")
    
    # Save detection counts to CSV
    with open(f"runs/obb/{run_name}/detection_counts.csv", 'w') as f:
        f.write("Frame,DetectionCount\n")
        for i, count in enumerate(detection_counts):
            f.write(f"{i},{count}\n")
    
    # Summary statistics
    total_frames = len(detection_counts)
    frames_with_detections = sum(1 for count in detection_counts if count > 0)
    total_detections = sum(detection_counts)
    
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with detections: {frames_with_detections} ({frames_with_detections/total_frames*100:.2f}%)")
    print(f"Total objects detected: {total_detections}")
    print(f"Average detections per frame: {total_detections/total_frames:.2f}")