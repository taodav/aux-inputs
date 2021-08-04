import numpy as np
from pathlib import Path
from PIL import Image


def save_info(results_path: Path, info: dict):
    np.save(results_path, info)


def load_info(results_path: Path):
    return np.load(results_path, allow_pickle=True).item()


def save_gif(arr: np.ndarray, path: Path, duration=400):
    gif = [Image.fromarray(img) for img in arr]

    gif[0].save(path, save_all=True, append_images=gif[1:], duration=duration, loop=0)


def save_video(arr: np.ndarray, path:Path, fps: int = 2):
    import cv2

    length, h, w, c = arr.shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

    for i in range(length * fps):
        frame = cv2.cvtColor(arr[i // fps], cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()




