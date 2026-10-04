import tempfile
import unittest
from pathlib import Path

from PIL import Image


class ImageTests(unittest.TestCase):
    def test_caps_image_area_and_keeps_rgb_aspect_ratio(self):
        from floodnet_rcmtd.ease2026.runner import bounded_image
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            Image.new("RGB", (1000, 500)).save(path)
            image = bounded_image(path, 262144)
            self.assertLessEqual(image.width * image.height, 262144)
            self.assertAlmostEqual(image.width / image.height, 2.0, delta=0.01)
            self.assertEqual(image.mode, "RGB")
