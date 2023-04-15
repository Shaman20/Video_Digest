import cv2
import os

frame_folder_path = 'C:\\Users\\shama\\college-project\\output\\1'

output_file_path = 'C:\\Users\\shama\\college-project'

frame_rate = 30.0
frame_size = (640, 480)

frame_files = [f for f in os.listdir(frame_folder_path) if f.endswith('.jpg')]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file_path, fourcc, frame_rate, frame_size)
for frame_file in frame_files:
    frame_path = os.path.join(frame_folder_path, frame_file)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)
video_writer.release()
cv2.destroyAllWindows()
