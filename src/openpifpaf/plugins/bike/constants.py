import numpy as np

COCO_BICYCLE_SKELETON = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 2), (1, 3), (6, 4),
]

COCO_BICYCLE_KEYPOINTS = [
    'back_tire_end',      # 1
    'back_tire_center',   # 2
    'seat',               # 3
    'handle',             # 4
    'front_tire_center',  # 5
    'front_tire_end',     # 6
]


COCO_BICYCLE_UPRIGHT_POSE = np.array([
        [-1.40092975, -0.70604985, 2.0], # back_tire_end     #1
        [-0.89150077, -0.70604985, 2.0], # back_tire_center  #2
        [-0.47758965,  1.31743374, 2.0], # seat              #3
        [ 0.45636341,  1.50676566, 2.0], # handle            #4
        [ 0.90211387, -0.70604985, 2.0], # front_tire_center #5
        [ 1.4115429 , -0.70604985, 2.0], # front_tire_end    #6
])



HFLIP = {
    'back_tire_end': 'back_tire_end',
    'back_tire_center': 'back_tire_center',
    'seat': 'seat',
    'handle': 'handle',
    'front_tire_center': 'front_tire_center',
    'front_tire_end': 'front_tire_end',
}



COCO_BICYCLE_SIGMAS = [
    0.089,  # back_tire_end
    0.089,  # back_tire_center
    0.089,  # seat
    0.089,  # handle
    0.089,  # front_tire_center
    0.089,  # front_tire_end
]


COCO_BICYCLE_SCORE_WEIGHTS = [1.0] * len(COCO_BICYCLE_KEYPOINTS)


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]


def draw_skeletons(pose):
    import openpifpaf  # pylint: disable=import-outside-toplevel
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter()

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    ann = openpifpaf.Annotation(keypoints=COCO_BICYCLE_KEYPOINTS,
                                skeleton=COCO_BICYCLE_SKELETON,
                                score_weights=COCO_BICYCLE_SCORE_WEIGHTS)
    ann.set(pose, np.array(COCO_BICYCLE_SIGMAS) * scale)
    with openpifpaf.show.Canvas.annotation(
            ann, filename='docs/skeleton_bicycle_coco.png') as ax:
        keypoint_painter.annotation(ax, ann)




def print_associations():
    for j1, j2 in COCO_BICYCLE_SKELETON:
        print(COCO_BICYCLE_KEYPOINTS[j1 - 1], '-', COCO_BICYCLE_KEYPOINTS[j2 - 1])


if __name__ == '__main__':
    print_associations()

    # c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    # rotate = np.array(((c, -s), (s, c)))
    # rotated_pose = np.copy(COCO_DAVINCI_POSE)
    # rotated_pose[:, :2] = np.einsum('ij,kj->ki', rotate, rotated_pose[:, :2])
    draw_skeletons(COCO_BICYCLE_UPRIGHT_POSE)
