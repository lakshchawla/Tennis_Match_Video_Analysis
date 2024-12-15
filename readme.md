# Tennis Match Video Analysis

![Demo](demo.gif)

This project is a Python-based tool for analyzing tennis match videos. It leverages machine learning models to detect and track players, track the ball, and identify court keypoints, producing an annotated video as the final output.

---

## Features

- **Player Detection and Tracking**: Identifies players in video frames and tracks their movements throughout the match.
- **Ball Detection and Tracking**: Locates the tennis ball in each frame and smooths its trajectory through interpolation.
- **Court Keypoint Detection**: Extracts keypoints of the tennis court to assist in determining player positions and interactions with the court.
- **Video Annotation**: Combines all detections into a visually annotated video with bounding boxes and keypoints for better analysis.

---

## How It Works

The analysis pipeline involves the following steps:

1. **Court Keypoint Detection**  
   Extracts the keypoints of the tennis court from video frames, which provides a reference for identifying player and ball positions.

2. **Player Detection and Tracking**  
   Detects players in the video and tracks their movements across frames. Only the relevant players are chosen based on their positions relative to the court.

3. **Ball Detection and Tracking**  
   Tracks the tennis ball across frames and applies interpolation to fill in any gaps in detection, ensuring smooth tracking.

4. **Video Annotation**  
   The processed video frames are annotated with:
   - Bounding boxes for players and the ball.
   - Marked keypoints for the court.
   The final annotated video provides a visual representation of the analysis.

---

## Getting Started

### Prerequisites

- **Python Version**: Python 3.7 or higher.
- **Dependencies**: Install the required Python libraries listed in the `requirements.txt` file.

### Running the Project

1. Clone the repository to your local machine.  
2. Prepare the input video for analysis.  
3. Run the analysis script to generate the annotated video.

---

## Output

- **Annotated Video**:  
  The final output video includes:
  - Bounding boxes highlighting player movements and ball trajectory.
  - Court keypoints overlaid on the video for spatial context.

---

## Project Structure

- **Models**: Pre-trained models for detecting players, the ball, and court keypoints.
- **Scripts**: Includes the main analysis script and utility functions.
- **Tracker Stubs**: Cached data for faster testing and debugging.
- **Demo File**: A sample `demo.gif` showcasing the project output.

