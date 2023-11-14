import os
import shutil
import server.Config as Config
import logging
from _thread import *

import pycolmap
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction, visualization, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, localize_from_image

logger = logging.getLogger("hloc")
logger.setLevel(logging.DEBUG)

"""
[data structure]

datasets
    map1
        mapping - mapping images
            20221103_125848.jpg
            20221103_125848.jpg
            ...
        user_id_0 - user_0's query image
            20221103_125848.jpg
        user_id_1
            20221103_125848.jpg
        ...
    map2
        mapping
            20221103_125848.jpg
            ...
        user_id_0
            20221103_125848.jpg
            ...
        ...
    ...
"""

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HERE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = Path(__file__).parent


class MappingData:
    def __init__(self, map_name: str, 
                       feature_conf: str='superpoint_aachen', 
                       matcher_conf: str='superglue', 
                       retrieval_conf: str='netvlad'):
        self.map_name = map_name
        self.images = Path(BASE_PATH / f'datasets/{map_name}')
        # self.query_images = Path(BASE_PATH / f'datasets/{map_name}/query')
        self.outputs = Path(BASE_PATH / f'outputs/{map_name}-10pair') # pair num can be changed
        self.sfm_pairs = self.outputs / 'pairs-sfm.txt'
        self.sfm_dir = self.outputs / 'sfm'
        self.loc_pairs = self.outputs / 'pairs-loc.txt'
        self.feature_conf = extract_features.confs[feature_conf]
        self.matcher_conf = match_features.confs[matcher_conf]
        self.retrieval_conf = extract_features.confs[retrieval_conf]
        self.feature_path = self.outputs / f'{self.feature_conf["output"]}.h5'
        self.match_path = self.outputs / f'{self.feature_conf["output"]}_{self.matcher_conf["output"]}.h5'
        self.model = pycolmap.Reconstruction()
        self.model.read_binary(self.sfm_dir.as_posix())
        # self.references = [p.relative_to(self.images).as_posix() for p in (self.images / 'mapping').iterdir()]
        # self.ref_ids = [self.model.find_image_with_name(r).image_id for r in self.references]
        self.conf = {
            'estimation': {'ransac': {'max_error': 12}},
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }
        self.localizer = QueryLocalizer(self.model, self.conf)
    
    def localize(self, user_id: str, query_image: str):
        query_path = self.query_images / f'{user_id}/{query_image}'
        query_image = pycolmap.read_image(query_path.as_posix())
        query_image = self.localizer.preprocess_image(query_image)
        query_features = self.localizer.extract_features(query_image)
        query_features = self.localizer.match_features(query_features)
        query_features = self.localizer.retrieval(query_features)
        query_features = self.localizer.cluster(query_features)
        query_features = self.localizer.pose_from_cluster(query_features)

        query = 'query/20221103_130102.jpg'
        result, log = localize_sfm.localize_from_image(
            sfm_dir, 
            images, 
            query,
            loc_pairs,
            feature_path,
            match_path,
        )

        return query_features


map1 = MappingData('hangwon_park_wide')
map2 = MappingData('fastfive')
# # map1 - hangwon_park_wide
# images = Path(BASE_PATH / 'datasets/hangwon_park_wide')
# query_images = Path(BASE_PATH / 'datasets/hangwon_park_wide/query')
# outputs = Path(BASE_PATH / 'outputs/hangwon_park_wide_10pair')

# sfm_pairs = outputs / 'pairs-sfm.txt'
# sfm_dir = outputs / 'sfm'
# loc_pairs = outputs / 'pairs-loc.txt'

# feature_conf = extract_features.confs['superpoint_aachen']
# matcher_conf = match_features.confs['superglue']
# retrieval_conf = extract_features.confs['netvlad']

# feature_path = outputs / f'{feature_conf["output"]}.h5'
# match_path = outputs / f'{feature_conf["output"]}_{matcher_conf["output"]}.h5'

# model = pycolmap.Reconstruction()
# model.read_binary(sfm_dir.as_posix())

# references = [p.relative_to(images).as_posix() for p in (images / 'mapping').iterdir()]
# ref_ids = [model.find_image_with_name(r).image_id for r in references]

# conf = {
#     'estimation': {'ransac': {'max_error': 12}},
#     'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
# }
# localizer = QueryLocalizer(model, conf)