import unittest

import numpy as np

from coordinate_transform import ImageTransform


class ImageTransformTests(unittest.TestCase):
    def test_original_size_coordinates_are_unchanged(self):
        transform = ImageTransform.from_sizes((800, 600), (800, 600))
        points = np.asarray([[0, 0], [120.5, 250.25], [800, 600]], dtype=np.float32)
        np.testing.assert_allclose(transform.map_points_to_original(points), points)

    def test_upscaled_image_maps_back(self):
        transform = ImageTransform.from_sizes((320, 240), (640, 480))
        np.testing.assert_allclose(transform.map_points_to_original([[160, 120], [640, 480]]),
                                   [[80, 60], [320, 240]])

    def test_non_square_scale_uses_actual_independent_factors(self):
        transform = ImageTransform.from_sizes((320, 200), (480, 400))
        self.assertAlmostEqual(transform.scale_x, 1.5)
        self.assertAlmostEqual(transform.scale_y, 2.0)
        np.testing.assert_allclose(transform.map_points_to_original([[480, 400]]), [[320, 200]])

    def test_crop_and_padding_offsets_are_inverted(self):
        transform = ImageTransform.from_sizes((100, 80), (120, 100),
                                              crop=(10, 5, 10, 5), pad=(5, 7, 5, 7))
        # Crop content is 80x70; after padding the resized content is 110x86.
        processed = [[5, 7], [115, 93]]
        np.testing.assert_allclose(transform.map_points_to_original(processed), [[10, 5], [90, 75]])

    def test_boundary_coordinates_are_clipped(self):
        transform = ImageTransform.from_sizes((100, 50), (200, 100))
        np.testing.assert_allclose(transform.map_points_to_original([[-20, -4], [240, 120]]),
                                   [[0, 0], [100, 50]])

    def test_polygon_and_box_mapping(self):
        transform = ImageTransform.from_sizes((200, 100), (400, 300))
        polygon = transform.map_points_to_original([[20, 30], [200, 30], [200, 180], [20, 180]])
        np.testing.assert_allclose(polygon, [[10, 10], [100, 10], [100, 60], [10, 60]])
        np.testing.assert_allclose(transform.map_box_to_original([20, 30, 200, 180]), [10, 10, 100, 60])


if __name__ == "__main__":
    unittest.main()
