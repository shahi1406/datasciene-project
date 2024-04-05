import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('road_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    cv2.fillPoly(mask, [np.array(region_of_interest_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough line transform
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Display the result
    cv2.imshow('Lane Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
