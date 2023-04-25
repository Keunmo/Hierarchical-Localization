# dimension: 1280x960
# codec: H.264
# framerate: 30
# bitrate: 1235kbps


import cv2
import os
# import ffmpeg
from pathlib import Path


# def check_rotation(video: Path):
#     # this returns meta-data of the video file in form of a dictionary
#     meta_dict = ffmpeg.probe(str(video.absolute()))

#     # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
#     # we are looking for
#     rotateCode = None
#     if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
#         rotateCode = cv2.ROTATE_90_CLOCKWISE
#     elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
#         rotateCode = cv2.ROTATE_180
#     elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
#         rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
#     return rotateCode


def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode)


def video_capture(video: Path, output: Path, frameintv: int = 30):
    # rotateCode = check_rotation(video)
    rotateCode = cv2.ROTATE_180  # 그냥 영상 보고 고정.
    video = str(video.absolute())
    cam = cv2.VideoCapture(video)
    if not os.path.exists(output):
        os.makedirs(output)
    # frame
    currentframe = 0
    idx = 0
    while(True):
        ret,frame = cam.read()
        if not ret:
            break
        # if rotateCode is not None:
        #     frame = correct_rotation(frame, rotateCode)
        frame = correct_rotation(frame, rotateCode)
        name = output / '{}.jpg'.format(idx)
        idx+=1
        name = str(name.absolute())
        print ('Create ' + name)
        cv2.imwrite(name, frame)
        currentframe += frameintv
        cam.set(1, currentframe)
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = Path("20221123_161253.mp4")
    output = Path('videoCapture')
    frameintv = 5
    video_capture(video, output, frameintv)
