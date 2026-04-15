# Video commands

## FFmpeg (FFprobe)

1. 看视频帧数：ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 <xxx.mp4>