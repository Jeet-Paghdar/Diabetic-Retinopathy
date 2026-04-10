"""
test_validation.py — Quick test for retinal image validation
Run: python tests/test_validation.py
"""
import sys
import os
import numpy as np
import cv2

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import is_retinal_image


def make_retinal_like():
    """Create a synthetic retinal-like image: red-dominant circle on black."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Draw a filled circle (bright red/orange center)
    cv2.circle(img, (112, 112), 90, (30, 60, 160), -1)  # BGR: low B, med G, high R
    # Add some texture
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def make_random_noise():
    """Create a random noise image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


def make_blue_image():
    """Create a solid blue image (definitely not retinal)."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # BGR: high blue
    img[:, :, 1] = 50
    img[:, :, 2] = 30
    return img


def make_landscape():
    """Create a green/blue landscape-like image."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Sky (top half)
    img[:112, :, 0] = 180  # Blue
    img[:112, :, 1] = 140
    img[:112, :, 2] = 100
    # Grass (bottom half)
    img[112:, :, 0] = 30
    img[112:, :, 1] = 150  # Green
    img[112:, :, 2] = 50
    # Add noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def test_real_retinal_image():
    """Test with actual raw retinal images from the dataset if available."""
    # Use RAW images — preprocessed ones lose color properties (Ben Graham normalization)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train_images')
    if not os.path.exists(data_dir):
        return None
    images = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
    if not images:
        return None
    img_path = os.path.join(data_dir, images[0])
    img = cv2.imread(img_path)
    if img is None:
        return None
    return is_retinal_image(img)


def main():
    print("=" * 60)
    print("  RETINAL IMAGE VALIDATION — TEST SUITE")
    print("=" * 60)

    tests = [
        ("Synthetic retinal-like (red circle on black)", make_retinal_like(), True),
        ("Random noise image", make_random_noise(), False),
        ("Solid blue image", make_blue_image(), False),
        ("Landscape (green/blue)", make_landscape(), False),
    ]

    passed = 0
    failed = 0

    for name, img, expected_valid in tests:
        is_valid, conf, reason, state = is_retinal_image(img)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n  Test: {name}")
        print(f"    Expected: {'VALID' if expected_valid else 'INVALID'}")
        print(f"    Got:      {'VALID' if is_valid else 'INVALID'} (conf={conf:.2f})")
        print(f"    Reason:   {reason}")
        print(f"    Result:   [{status}]")

    # Test with real retinal image if available
    real_result = test_real_retinal_image()
    if real_result:
        is_valid, conf, reason, state = real_result
        expected = True
        status = "PASS" if is_valid == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"\n  Test: Real retinal image from dataset")
        print(f"    Expected: VALID")
        print(f"    Got:      {'VALID' if is_valid else 'INVALID'} (conf={conf:.2f})")
        print(f"    Reason:   {reason}")
        print(f"    Result:   [{status}]")
    else:
        print(f"\n  [SKIP] No real retinal images found in data/processed/train_images/")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
