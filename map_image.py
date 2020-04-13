import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


class MapImage:

    def __init__(self, coordinates=None, image_id=None, image_path=None,
                 image_type=None, image_dim=None):

        self.coordinates = coordinates
        self.image_id = image_id
        self.image_path = image_path

        if (not image_path) and (not image_dim):
            raise Exception('Provide either an image_path or '
                            'image_dimensions (width, height) in pixels')

        self.image = None
        self.image_type = None

        self.has_been_preprocessed = False
        self.has_been_normalized = False

        if self.image_path:
            # load image if path was provided
            self.image = self._load_image()
            if image_type:
                self.image_type = image_type
            else:
                self.image_type = self._get_image_type()
        else:
            # create a black image with given dimensions
            self.image = np.zeros((image_dim[0], image_dim[1], 3),
                                  dtype=np.uint8)

        # keep track of how noisy the original image was
        # self.percentage_noisy_pixels = None

    def print_info(self):

        output_strings = [
            f'[res: {int(self.px_per_lon_ratio)} x {int(self.px_per_lat_ratio)}]',
            f'[size: {self.width_px} x {self.height_px}]',
            f'[type: {self.image_type}]',
            f'[center: {self.central_point}]',
        ]
        output_str = ''.join(output_strings)
        print(output_str)

    def _load_image(self):
        """
        Returns RGB image
        """
        image = cv2.imread(self.image_path, 3)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess_image_old(self):

        # to gray-scale
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # blur (supposed to delete straight lines)
        self.image = cv2.medianBlur(self.image, ksize=35)
        ret, self.image = cv2.threshold(self.image, thresh=185, maxval=255,
                                        type=cv2.THRESH_BINARY)
        self.image = np.repeat(self.image[:, :, np.newaxis], 3, axis=2)

    def preprocess_image_new(self):

        if not self.has_been_preprocessed:
            if self.image_type == 'blue-green':
                self._detect_and_remove_noise()
                self._set_known_land_pixels_to_black()

            # to gray-scale
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # blur (supposed to delete straight lines)
            self.image = cv2.medianBlur(self.image, ksize=45)
            ret, self.image = cv2.threshold(self.image, thresh=185, maxval=255,
                                            type=cv2.THRESH_BINARY)
            self.image = np.repeat(self.image[:, :, np.newaxis], 3, axis=2)

            self.has_been_preprocessed = True

    def normalize_pixel_range(self):
        if not self.has_been_normalized:
            self.image = self.image / 255
            self.has_been_normalized = True

    def _set_known_land_pixels_to_black(self):

        land_colors_rgb = [
            {'color': [216, 208, 195], 'diff': 20},
            {'color': [199, 211, 209], 'diff': 15},
            {'color': [226, 232, 242], 'diff': 10},
            {'color': [169, 191, 199], 'diff': 10},
            {'color': [169, 201, 187], 'diff': 10},
            {'color': [189, 205, 186], 'diff': 10},
            # {'color': [182, 190, 209], 'diff': 10},
        ]
        for x in land_colors_rgb:
            diff = x['diff']
            color = x['color']
            lower = np.array([(color[i] - diff) for i in range(0, 3)],
                             dtype=np.uint8)
            upper = np.array([(color[i] + diff) for i in range(0, 3)],
                             dtype=np.uint8)
            mask = cv2.inRange(self.image, lower, upper)
            self.image[mask == 255] = [0, 0, 0]

    def _detect_and_remove_noise(self):

        # detect artificial straight lines and replace them with white
        hsv_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

        # saturation above 70 seems to work well
        low_H = 0
        high_H = 180
        low_S = 80  # 110?
        high_S = 255
        low_V = 0
        high_V = 255

        mask = cv2.inRange(hsv_img, (low_H, low_S, low_V),
                           (high_H, high_S, high_V))

        # replace noisy lines with white color
        # TODO: maybe improve this?
        self.image[mask == 255] = [255, 255, 255]
        # self.plot()

    def _get_image_type(self):
        """
        Image type in {'white-orange', 'white-gray', 'blue-green'}
        """
        # get percentage of white pixels in image
        ratio_white_pixels = \
            self._get_percentage_pixels_given_color([255, 255, 255])

        # get percentage of orange pixels in image
        ratio_gray_pixels = \
            self._get_percentage_pixels_given_color([128, 128, 128])

        ratio_orange_pixels = \
            self._get_percentage_pixels_given_color([244, 164, 96])

        #         print('Ratio white pixels: ', ratio_white_pixels)
        #         print('Ratio gray pixels: ', ratio_gray_pixels)
        #         print('Ratio orange pixels: ', ratio_orange_pixels)

        if (ratio_white_pixels > 0.20) and (ratio_gray_pixels > 0.05):
            image_type = 'white-gray'
        #         elif ratio_white_pixels > 0.20:
        #             image_type = 'white-orange'
        elif ratio_orange_pixels > 0.03:
            image_type = 'white-orange'
        elif ratio_gray_pixels > 0.05:
            image_type = 'white-gray'
        elif ratio_white_pixels > 0.50:
            image_type = 'white-gray'
        else:
            image_type = 'blue-green'

        return image_type

    def _get_percentage_pixels_given_color(self, color_rgb):
        """
        Given 'color_rgb' this function returns the percentage of image pixels
        of this color.
        """
        diff = 0
        boundaries = [
            ([color_rgb[0] - diff, color_rgb[1] - diff, color_rgb[2] - diff],
             [color_rgb[0] + diff, color_rgb[1] + diff, color_rgb[2] + diff])]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(self.image, lower, upper)
            ratio_color = cv2.countNonZero(mask) / (self.image.size / 3)

        return ratio_color

    # @property
    # def central_coordinates(self):
    #
    #     # edge case (near Australia)
    #
    #
    #     # normal case
    #     center = [0.5 * (self.ll_lon + self.ur_lon),
    #               0.5 * (self.ll_lat + self.ur_lat)]

    @property
    def central_point(self):
        if self.coordinates:
            center = [0.5 * (self.ll_lon + self.ur_lon),
                      0.5 * (self.ll_lat + self.ur_lat)]

            # beautify coordinates
            e_w = 'E'
            lon = center[0]
            if center[0] < 0:
                e_w = 'W'
                lon = -center[0]
            n_s = 'N'
            lat = center[1]
            if center[1] < 0:
                n_s = 'S'
                lat = -center[1]

            lon = round(lon, 1)
            lat = round(lat, 1)

            return f'{lat}°{n_s} {lon}°{e_w}'
        else:
            raise Exception('Map coordinates not specified')

    @property
    def ll_lon(self):
        if self.coordinates:
            return self.coordinates[0]
        else:
            raise Exception('Map coordinates not specified')

    @property
    def ll_lat(self):
        if self.coordinates:
            return self.coordinates[1]
        else:
            raise Exception('Map coordinates not specified')

    @property
    def ur_lon(self):
        if self.coordinates:
            return self.coordinates[2]
        else:
            raise Exception('Map coordinates not specified')

    @property
    def ur_lat(self):
        if self.coordinates:
            return self.coordinates[3]
        else:
            raise Exception('Map coordinates not specified')

    @property
    def px_per_lon_ratio(self):
        if self.coordinates:
            if self.ll_lon > self.ur_lon:
                # edge case that happens in Australia
                return self.width_px / (360 - self.ll_lon + self.ur_lon)
            else:
                return self.width_px / (self.ur_lon - self.ll_lon)

        else:
            raise Exception('Map coordinates not specified')

    @property
    def px_per_lat_ratio(self):
        if self.coordinates:
            return self.height_px / (self.ur_lat - self.ll_lat)
        else:
            raise Exception('Map coordinates not specified')

    @property
    def area(self):
        return abs((self.ur_lon - self.ll_lon) * (self.ur_lat - self.ll_lat))

    def plot(self):
        plt.imshow(self.image)
        plt.show()

    @property
    def shape(self):
        return self.image.shape

    @property
    def width_px(self):
        return self.image.shape[1]

    @property
    def height_px(self):
        return self.image.shape[0]

    def resize_by_scaling_factor(self, scaling_factor):
        """
        """
        new_height = int(scaling_factor * self.height_px)
        new_width = int(scaling_factor * self.width_px)
        new_dim = (new_width, new_height)
        self.image = cv2.resize(self.image, new_dim,
                                interpolation=cv2.INTER_AREA)

    def plot_hsv_3d_scatterplot(self):

        img = self.image.copy()

        pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        fig = plt.figure(figsize=(20, 10))
        axis = fig.add_subplot(1, 2, 1, projection="3d")
        axis.scatter(h.flatten(), v.flatten(), s.flatten(),
                     facecolors=pixel_colors,
                     marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Value")
        axis.set_zlabel("Saturation")

        axis = fig.add_subplot(1, 2, 2, projection="3d")
        axis.scatter(s.flatten(), h.flatten(), v.flatten(),
                     facecolors=pixel_colors,
                     marker=".")
        axis.set_xlabel("Saturation")
        axis.set_ylabel("Hue")
        axis.set_zlabel("Value")

        plt.show()