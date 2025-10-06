import cv2
import numpy as np

# File paths (modifica i percorsi se necessario)
input_video_path = 'Multiple View.avi'
reference_frame_path = 'ReferenceFrame.png'
object_mask_path = 'ObjectMask.png'
augmented_layer_path = 'AugmentedLayer.png'
augmented_mask_path = 'AugmentedLayerMask.png'
output_video_path = 'Augmented Multiple View1.avi'

# Carica immagini e maschere
reference_frame = cv2.imread(reference_frame_path)
object_mask = cv2.imread(object_mask_path, 0)
augmented_layer = cv2.imread(augmented_layer_path)
augmented_mask = cv2.imread(augmented_mask_path, 0)

# Inizializza il video
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Writer per il video di output
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Rileva i keypoints nel frame di riferimento (usiamo ORB per semplicità)
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01, edgeThreshold=10)
#orb = cv2.ORB_create(1000)
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_frame, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Trova i keypoints nel frame corrente
    keypoints_frame, descriptors_frame = sift.detectAndCompute(frame, None)

    # Match tra descrittori
    if descriptors_frame is not None and len(descriptors_frame) >= 4:
        matches = bf.match(descriptors_ref, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) >= 4:
            src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Trova l'omografia
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Warpa il livello AR nella vista corrente
                warped_layer = cv2.warpPerspective(augmented_layer, H, (frame.shape[1], frame.shape[0]))
                warped_mask = cv2.warpPerspective(augmented_mask, H, (frame.shape[1], frame.shape[0]))

                mask_inv = cv2.bitwise_not(warped_mask)
                background = cv2.bitwise_and(frame, frame, mask=mask_inv)
                foreground = cv2.bitwise_and(warped_layer, warped_layer, mask=warped_mask)

                frame = cv2.add(background, foreground) #Merges the two images to create the augmented frame.
                cv2.imshow('back', background)
    out.write(frame)  #Save the frame in output video.

    
cap.release()      
out.release()
cv2.destroyAllWindows()
