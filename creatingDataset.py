import os
import cv2

def extract_frames(video_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = 1
    # Loop through each video file in the input folder
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
            # Open the video file
            video_path = os.path.join(video_folder, filename)
            cap = cv2.VideoCapture(video_path)

            # Get the video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a video writer to save the frames
            output_path = os.path.join(output_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Loop through the video and extract a frame every 3 seconds
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % (int(fps) * 3) == 0:
                    #output_file = os.path.join(output_path, f"frame_{frame_count//int(fps)}.png")
                    cv2.imwrite(f"{output_folder}/{f}.png", frame)
                    f=f+1
                frame_count += 1

            # Release the video capture object
            cap.release()

# Example usage
extract_frames("dataset/unseen_test/input_videos", "dataset/unseen_test/output_frames")