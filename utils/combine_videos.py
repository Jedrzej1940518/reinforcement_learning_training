import json
import os
from glob import glob
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, ImageClip
import moviepy.video.fx.all as vfx

from PIL import Image, ImageDraw, ImageFont


directory = "repos/reinforcement_learning/DQN/Breakout/videos"
output_dir = "repos/reinforcement_learning/DQN/Breakout"
steps_total = 2_000_000

def create_watermark(metadata, size=(200, 40), text_color=(255, 255, 255, 255), bg_color=(0, 0, 0, 128)):
    img = Image.new('RGBA', size, bg_color)
    d = ImageDraw.Draw(img)
    # Using a default font; consider adjusting the size if the text appears too small or too large
    font = ImageFont.truetype("arial.ttf", 10)
    metadata_str = ", ".join(f"{key}: {value}" for key, value in metadata.items())
    #{"step_id": 55706, "episode_id": 150, "content_type": "video/mp4"}
    episode = metadata["episode_id"]
    steps = metadata["step_id"] / 2_000_000 * 100
    metadata_str = f"episode: {episode}, progress: {steps:.1f}%"
    #metadata_str = "whats up brother"  # For testing purposes
    d.text((40, 20), metadata_str, fill=text_color, font=font)
    return img

episode_number_pattern = re.compile(r"episode-(\d+)")

video_files = glob(os.path.join(directory, '*.mp4'))
processed_clips = []
watermark_paths = []  # To keep track of watermark image paths for cleanup

sorted_video_files = sorted(video_files, key=lambda x: int(episode_number_pattern.search(x).group(1)))

for video_file in sorted_video_files:
    metadata_file = video_file.replace('.mp4', '.meta.json')
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    clip = VideoFileClip(video_file)
    clip = clip.fx(vfx.speedx, 1.25)  # Speed up by a factor of 2

    watermark_img = create_watermark(metadata)
    watermark_img_path = video_file.replace('.mp4', '_watermark.png')
    watermark_img.save(watermark_img_path)
    watermark_paths.append(watermark_img_path)  # Add the path for later cleanup
    
    watermark_clip = ImageClip(watermark_img_path).set_duration(clip.duration).set_position(("right", "top")).set_opacity(1)
    
    video_with_watermark = CompositeVideoClip([clip, watermark_clip])
    
    processed_clips.append(video_with_watermark)

final_clip = concatenate_videoclips(processed_clips)

output_file = os.path.join(output_dir, "gameplay.mp4")
final_clip.write_videofile(output_file, codec="mpeg4", audio_codec="aac")

# Cleanup: Delete the watermark images
for watermark_path in watermark_paths:
    os.remove(watermark_path)
