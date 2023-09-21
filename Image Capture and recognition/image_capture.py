import cv2
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to a file
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and saved as 'captured_image.jpg'")

        cv2.imshow("Me",frame)

        cv2.waitKey(0)
        cv2.destroyWindow("Me")


    # Release the camera
    #cap.release()

# Close any open windows
#cv2.destroyAllWindows()
