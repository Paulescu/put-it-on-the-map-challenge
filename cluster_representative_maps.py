import random
import numpy as np
import math
from map_image import MapImage


class ClusterRepresentativeMaps:

    def __init__(self, n_map_clusters, n_maps_per_cluster, map_id_to_metadata,
                 map_class_to_image_ids, map_id_to_path,
                 map_id_to_actual_coordinates,
                 resize_factor=1):

        self.n_map_clusters = n_map_clusters
        self.n_maps_per_cluster = n_maps_per_cluster
        self.map_id_to_metadata = map_id_to_metadata
        self.map_class_to_image_ids = map_class_to_image_ids

        self.map_id_to_path = map_id_to_path
        self.map_id_to_actual_coordinates = map_id_to_actual_coordinates

        self.resize_factor = resize_factor

        self.map_class_to_map_ids = None
        self.map_class_to_map_images = None

        self.map_class_to_min_width_px = None
        self.map_class_to_min_height_px = None

    def generate_map_class_to_map_ids(self):
        """
        """
        self.map_class_to_map_ids = dict()

        for cluster in range(0, self.n_map_clusters):

            image_ids = \
                self._get_ids_maps_with_diverse_resolution_and_distance_to_center(
                    cluster)

            # add random ones if necessary until we have 'n_maps_per_cluster'
            n_missing_elements = self.n_maps_per_cluster - len(image_ids)
            if n_missing_elements > 0:
                other_ids = list(
                    set(self.map_class_to_image_ids[cluster]) - set(image_ids))
                other_ids = random.sample(other_ids, n_missing_elements)
                image_ids += other_ids

            self.map_class_to_map_ids[cluster] = image_ids
            print(f'Cluster {cluster}, {len(image_ids)} maps')

    def generate_map_class_to_map_images(self):
        """
        """
        if not self.map_class_to_map_ids:
            self.generate_map_class_to_map_ids()

        self.map_class_to_map_images = dict()
        self.map_class_to_min_width_px = dict()
        self.map_class_to_min_height_px = dict()

        for cluster in range(0, self.n_map_clusters):
            map_images = list()
            for image_id in self.map_class_to_map_ids[cluster]:
                map_image = MapImage(image_id=image_id,
                                     image_path=self.map_id_to_path[image_id],
                                     coordinates=
                                     self.map_id_to_actual_coordinates[
                                         image_id])
                map_image.preprocess_image_new()

                # e.g. to decrease resolution and speed up template matching
                map_image.resize_by_scaling_factor(self.resize_factor)

                map_images.append(map_image)

            # compute min width/height
            self.map_class_to_min_width_px[cluster] = \
                min([m.width_px for m in map_images])
            self.map_class_to_min_height_px[cluster] = \
                min([m.height_px for m in map_images])

            self.map_class_to_map_images[cluster] = map_images
            print(f'Cluster {cluster}, {len(map_images)} maps')

    def get_min_width_px(self, cluster):
        return self.map_class_to_min_width_px[cluster]

    def get_min_height_px(self, cluster):
        return self.map_class_to_min_height_px[cluster]

    def get_map_representatives(self, cluster):
        return self.map_class_to_map_images[cluster]

    def _get_ids_maps_with_diverse_resolution_and_distance_to_center(self,
                                                                     cluster):
        """
        Finds representative maps for each cluster. It does so, by trying to
        cover the whole resolution space and different distances to the cluster
        center.

        TODO: new version that partitions space of maps inside a cluster according
        to the following 3 dimenstions:
        - distance_to_center (polar coordinate 1)
        - degrees_from_center (polar coordinate 2)
        - resolution
        """
        # list of image_ids for this cluster, for which we have
        # metadata
        image_ids = self.map_class_to_image_ids[cluster]
        ids_with_metadata = [i for i in self.map_id_to_metadata.keys()]
        image_ids = list(set(image_ids) & set(ids_with_metadata))

        # compute range of resolution
        # enough to work with horizontal resolution (h and v are highly correlated)
        resolutions = [self.map_id_to_metadata[i]['px_per_lon_ratio'] for i in
                       image_ids]
        res_min = min(resolutions)
        res_max = max(resolutions)
        res_num_steps = int(math.sqrt(self.n_maps_per_cluster))
        res_step = (res_max - res_min) / res_num_steps
        # print('Resolution')
        # print(f'min: {res_min}, max: {res_max}, step: {res_step}')

        # compute range of distance to the cluster center
        distances = [self.map_id_to_metadata[i]['distance_to_center'] for i in
                     image_ids]
        d_min = min(distances)
        d_max = max(distances)
        d_num_steps = int(math.sqrt(self.n_maps_per_cluster))
        d_step = (d_max - d_min) / d_num_steps
        # print('Distance to center')
        # print(f'min: {d_min}, max: {d_max}, step: {d_step}')

        # parameter (no reason to change if from 1)
        max_maps_per_bucket = 1

        n_maps_per_bucket = np.zeros((res_num_steps, d_num_steps),
                                     dtype=np.uint32)
        # print(n_maps_per_bucket)

        selected_image_ids = list()
        for image_id in image_ids:
            # resolution
            res = self.map_id_to_metadata[image_id]['px_per_lon_ratio'] - 0.01
            d = self.map_id_to_metadata[image_id]['distance_to_center'] - 0.01

            r_bucket = int((res - res_min) / res_step)
            d_bucket = int((d - d_min) / d_step)
            # print('res: ', res, r_bucket)
            # print('d: ', d, d_bucket)
            n_maps_in_this_bucket = n_maps_per_bucket[r_bucket, d_bucket]

            if n_maps_in_this_bucket < max_maps_per_bucket:
                # pick this map
                selected_image_ids.append(image_id)

                # update counter
                n_maps_per_bucket[r_bucket, d_bucket] += 1
                # print(n_maps_per_bucket)

        return selected_image_ids