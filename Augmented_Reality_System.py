import cv2
import numpy as np
from pathlib import Path

# Base directory for the project files
BASE = Path("")  

# Input and output file paths
input_video_path = BASE / 'Multiple View.avi'
reference_frame_path = BASE / 'ReferenceFrame.png'
object_mask_path = BASE / 'ObjectMask.png'
augmented_layer_path = BASE / 'AugmentedLayer.png'
augmented_mask_path = BASE / 'AugmentedLayerMask.png'
output_video_path = BASE / 'Augmented Multiple View.avi'

# Load reference images and masks (0 flag loads masks in grayscale)
reference_frame = cv2.imread(reference_frame_path)
object_mask = cv2.imread(object_mask_path, 0)
augmented_layer = cv2.imread(augmented_layer_path)
augmented_mask = cv2.imread(augmented_mask_path, 0)

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer for the output file
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Initialize SIFT feature detector and compute reference descriptors
# Using the object_mask ensures features are only extracted from the book itself
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10)
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_frame, mask=object_mask)

# Initialize Brute Force Matcher with L2 norm and cross-check enabled for accuracy
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect features in the current video frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(frame, None)

    # Proceed if enough features are found
    if descriptors_frame is not None and len(descriptors_frame) >= 4:
        # Match current frame descriptors with reference descriptors
        matches = bf.match(descriptors_ref, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Ensure minimum number of matches required to compute homography
        if len(matches) >= 4:
            # Extract coordinates of matched keypoints
            src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Compute the homography matrix using RANSAC to filter outliers
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Warp the AR layer and the provided mask to match the current camera perspective
                warped_layer = cv2.warpPerspective(augmented_layer, H, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                warped_mask = cv2.warpPerspective(augmented_mask, H, (frame.shape[1], frame.shape[0]))
                
                
                # Isolate the bright white text and logo from the AR layer
                lower_white = np.array([200, 200, 200], dtype=np.uint8)
                upper_white = np.array([255, 255, 255], dtype=np.uint8)
                white_content_mask = cv2.inRange(augmented_layer, lower_white, upper_white)
                
                # Warp the white content mask to match the perspective
                warped_white_mask = cv2.warpPerspective(white_content_mask, H, (frame.shape[1], frame.shape[0]))
                
                # Initialize the alpha channel using the provided AR mask 
                alpha = warped_mask.astype(np.float32) / 255.0 
                
                # Identify the background area (inside the AR mask but excluding the white text/logo)
                blue_bg_condition = (alpha > 0.4) & (warped_white_mask < 80) 
                
                # Reduce opacity of the background area to 20% to allow physical shadows to blend naturally
                alpha[blue_bg_condition] = 0.20

                # Apply Gaussian blur to the alpha channel for anti-aliased edge blending
                alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
                alpha = cv2.merge([alpha, alpha, alpha])
                
                # Composite the warped AR layer onto the video frame using the alpha matte
                frame = cv2.convertScaleAbs(warped_layer * alpha + frame * (1.0 - alpha))
                
    # Write the composited frame to the output video
    out.write(frame)  

# Release resources and close windows
cap.release()      
out.release()
cv2.destroyAllWindows()
