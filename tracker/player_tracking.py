from ultralytics import YOLO 
import cv2
import pickle

from utils import calculate_player_centres, calculate_min_distance

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        
    def choose_players(self, kps, players):
        p1, p2 = self.filter_players(kps, players[0])
        filtered_player_detections = []
        
        for player_dicts in players:
            # print({p1: player_dicts[p1], p2: player_dicts[p2]})
            filtered_player_detections.append({p1: player_dicts[p1], p2: player_dicts[p2]})
        
        return filtered_player_detections
        # return {p1: players[p1], p2: players[p2]}

    def filter_players(self, kps, players):
        # print(type(players))
        # print(type(players[0]))
        player_center = calculate_player_centres(players)
        for i, p_nums in enumerate(list(player_center.keys())):
            player_center[p_nums] = float(calculate_min_distance(kps, player_center[p_nums]))
        
        player_center = dict(sorted(player_center.items(), key=lambda item: item[1]))
        p1, p2 = list(player_center.keys())[:2]
        return p1, p2
        # return {p1: players[p1], p2: players[p2]}
        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        # player_detections = self.filter_players(kps, player_detections)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            output_video_frames.append(frame)
            
        return output_video_frames