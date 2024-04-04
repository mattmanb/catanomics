import cv2

def capture_screenshot():
    # Start capturing video from the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    screenshot = None
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv2.imshow('Camera', frame)

            # Define the key check for 'y' and Escape
            key = cv2.waitKey(1) & 0xFF

            # Capture the screenshot when 'y' is pressed
            if key == ord('y'):
                screenshot = frame
                break
            
            # Break the loop if Escape is pressed
            if key == 27:  # 27 is the ASCII code for Escape key
                break

    except KeyboardInterrupt:
        # Handle any manual interrupts gracefully
        print("Streaming stopped.")
    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    return screenshot

# Example usage
screenshot = capture_screenshot()
if screenshot is not None:
    print("Screenshot captured.")
    # At this point, you can process the screenshot as needed,
    # For example, display it using cv2.imshow("Screenshot", screenshot)
    # or save it to a file with cv2.imwrite("screenshot.jpg", screenshot)
else:
    print("No screenshot captured.")
