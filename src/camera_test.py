import numpy as np
import unittest
import numpy as np

from datalink.data import Position, Rotation, Pose, ImageParams
from camera import UnrealEngineCamera, RPiv2Camera


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.pose = Pose(position=Position(0, 0.135, 0), rotation=Rotation(0, -15.05, 0))
        self.img_params = ImageParams(width=820, height=616, fov_deg=90)
        
        # Use UnrealEngineCamera for testing since Camera is abstract
        self.camera = UnrealEngineCamera(self.pose, self.img_params)
    
    def test_initialization(self):
        # Test that matrices and transforms are created correctly
        self.assertEqual(self.camera.pose.position.y, 0.135)
        self.assertEqual(self.camera.rotation.pitch, -15.05)
        self.assertTrue(hasattr(self.camera, 'M'))
        self.assertTrue(hasattr(self.camera, 'R_rc'))
        self.assertTrue(hasattr(self.camera, 'H_cr'))
        
    def test_coordinate_transforms(self):
        # Test camera to road transforms
        test_point_cam = np.array([1.0, 2.0, 3.0, 1])
        point_road = self.camera.cam2road(test_point_cam)
        point_cam_again = self.camera.road2cam(point_road)
        np.testing.assert_array_almost_equal(test_point_cam, point_cam_again)
    
    def test_uv_transforms(self):
        # Test image coordinates to 3D coordinates
        u, v = 400, 300
        xyz_cam = self.camera.uv2xyz_camframe(u, v)
        uv_again = self.camera.xyz_camframe2uv(xyz_cam)
        self.assertAlmostEqual(u, uv_again[0], delta=1)
        self.assertAlmostEqual(v, uv_again[1], delta=1)
        
    def test_roadframe_transforms(self):
        # Test image to road frame coordinates
        u, v = 400, 300
        xyzw_road = self.camera.uv2xyzw_roadframe(u, v)
        uv_again = self.camera.xyzw_roadframe2uv(xyzw_road)
        self.assertAlmostEqual(u, uv_again[0], delta=1)
        self.assertAlmostEqual(v, uv_again[1], delta=1)
    
    def test_iso8855_transforms(self):
        # Test ISO8855 coordinate transforms
        u, v = 400, 300
        xyz_iso8855 = self.camera.uv2xyz_roadframe_iso8855(u, v)
        uv_again = self.camera.xyz_roadframe_iso88552uv(xyz_iso8855)
        self.assertAlmostEqual(u, uv_again[0], delta=1)
        self.assertAlmostEqual(v, uv_again[1], delta=1)

class TestUnrealEngineCamera(unittest.TestCase):
    def setUp(self):
        self.pose = Pose(position=Position(0, 0.135, 0), rotation=Rotation(0, -15.05, 0))
        self.img_params = ImageParams(width=820, height=616, fov_deg=90)
        self.camera = UnrealEngineCamera(self.pose, self.img_params)
    
    def test_intrinsic_matrix(self):
        M = self.camera.intrinsic_matrix()
        # Check focal length calculation for 90 deg FOV
        expected_a = (self.img_params.width / 2.0) / np.tan(np.deg2rad(90) / 2.0)
        self.assertAlmostEqual(M[0, 0], expected_a)
        self.assertAlmostEqual(M[1, 1], expected_a)
        # Principal point should be at image center
        self.assertAlmostEqual(M[0, 2], self.img_params.width / 2)
        self.assertAlmostEqual(M[1, 2], self.img_params.height / 2)

class TestRPiv2Camera(unittest.TestCase):
    def setUp(self):
        self.pose = Pose(position=Position(0, 0.135, 0), rotation=Rotation(0, -15.05, 0))
        self.img_params = ImageParams(width=820, height=616, fov_deg=90)
        self.camera = RPiv2Camera(self.pose, self.img_params)
    
    def test_intrinsic_matrix(self):
        M = self.camera.intrinsic_matrix()
        # Check that matrix has expected properties
        self.assertGreater(M[0, 0], 0)  # fx should be positive
        self.assertGreater(M[1, 1], 0)  # fy should be positive
        self.assertEqual(M[0, 0], M[1, 1])  # fx should equal fy for square pixels
        # Principal point should be at image center
        self.assertAlmostEqual(M[0, 2], self.img_params.width / 2)
        self.assertAlmostEqual(M[1, 2], self.img_params.height / 2)



if __name__ == "__main__":
    unittest.main()
