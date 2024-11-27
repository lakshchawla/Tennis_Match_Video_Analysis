import numpy as np

from utils import capture_video, save_video
from tracker import PlayerTracker
from tracker import BallTracker
from tracker import CourtKpsDetector

def main(): #
    input_video_path = "input_videos/input_video.mp4"
    video_frames = capture_video(input_video_path)
    
    player_tracker = PlayerTracker(model_path="./yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/last_100.pt")
    kpts_detector = CourtKpsDetector(model_path="./kps_model.pth")
    
    court_keypoints = kpts_detector.predict(video_frames[0])
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")
    player_detection = player_tracker.choose_players(court_keypoints, player_detection)
    
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl")
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)
    
    video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)
    video_frames  = kpts_detector.draw_keypoints_on_video(video_frames, court_keypoints)
            
    save_video(video_frames, "output/output_video.avi")
    
if __name__ == "__main__":
    main()