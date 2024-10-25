import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            print("Need at least 2 images to stitch.")
            return None, []
        
        max_width = 1000  # Set your desired maximum width
        for i in range(len(all_images)):
            img = cv2.imread(all_images[i])
            height, width = img.shape[:2]
            if width > max_width:
                scaling_factor = max_width / width
                img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))
            all_images[i] = img  # Store the resized image back

        stitched_image = all_images[0]
        homography_matrix_list = []

        for i in range(1, len(all_images)):
            next_image = all_images[i]
            stitched_image, H = self.stitch_images(stitched_image, next_image)
            homography_matrix_list.append(H)

        return stitched_image, homography_matrix_list 

    def stitch_images(self, left_img, right_img):
        key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)
        good_matches = self.match_keypoint(descriptor1, descriptor2)

        src_pts = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        final_H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Calculate output size based on homography
        height_left, width_left = left_img.shape[:2]
        height_right, width_right = right_img.shape[:2]
        pts_left = np.float32([[0, 0], [0, height_left], [width_left, height_left], [width_left, 0]]).reshape(-1, 1, 2)
        pts_right = np.float32([[0, 0], [0, height_right], [width_right, height_right], [width_right, 0]]).reshape(-1, 1, 2)
        pts_right_transformed = cv2.perspectiveTransform(pts_right, final_H)

        # Find the bounding box of the stitched image
        all_pts = np.concatenate((pts_left, pts_right_transformed), axis=0)
        [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp the left image
        stitched_img = cv2.warpPerspective(left_img, H_translation.dot(final_H), (x_max - x_min, y_max - y_min))
        stitched_img[translation_dist[1]:height_right + translation_dist[1], translation_dist[0]:width_right + translation_dist[0]] = right_img

        return stitched_img, final_H

    def get_keypoint(self, left_img, right_img):
        l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
        key_points2, descriptor2 = sift.detectAndCompute(r_img, None)

        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoint(self, descriptor1, descriptor2):
        matcher = cv2.BFMatcher()
        matches = matcher.match(descriptor1, descriptor2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]  # Keep top 50 matches for robustness

        return good_matches
