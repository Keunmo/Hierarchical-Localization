from operator import mod
import os
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, match_dense
from hloc.visualization import plot_images, read_image
# from hloc.utils import viz_3d
import pycolmap


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target = 'MipNeRF360/bicycle/images'

images = Path('../data/' + target)
outputs = Path('../outputs/' + 'MipNeRF360/bicycle/231107_1421')

sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']
matcher_conf = match_dense.confs['loftr']

# references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
references = [p.relative_to(images).as_posix() for p in images.iterdir()]

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
# match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
features, matches = match_dense.main(matcher_conf, sfm_pairs, images, outputs, max_kps=8192, overwrite=False)

model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
