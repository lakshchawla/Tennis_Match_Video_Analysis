import cv2 as cv

# function for reading and capturing frames from the vdeo file
def capture_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames    

def save_video(output_video_frames, output_video_path):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(output_video_path, fourcc, 30.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()