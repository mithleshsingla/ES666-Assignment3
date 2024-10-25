import pdb
import glob
import cv2
import os
import numpy as np
import random
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func

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
        
        max_width = 2000  # Set your desired maximum width
        for i in range(len(all_images)):
            img = cv2.imread(all_images[i])
            height, width = img.shape[:2]
            if width > max_width:
                scaling_factor = max_width / width
                img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))
            all_images[i] = img  # Store the resized image back


        # Start stitching from the first image
        stitched_image = all_images[0]
        homography_matrix_list = []

        for i in range(1, len(all_images)):
            next_image = all_images[i]
            stitched_image, H = self.stitch_images(stitched_image, next_image)
            homography_matrix_list.append(H)

        return stitched_image, homography_matrix_list 

    def stitch_images(self, left_img, right_img):
        # Get keypoints and descriptors
        key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)
        good_matches = self.match_keypoint(key_points1, key_points2, descriptor1, descriptor2)

        # Estimate homography using RANSAC
        final_H = self.ransac(good_matches)

        # Stitch the images together using the homography
        result_img = self.apply_homography(left_img, right_img, final_H)

        return result_img, final_H

    def apply_homography(self, left_img, right_img, H):
        rows1, cols1 = left_img.shape[:2]
        rows2, cols2 = right_img.shape[:2]

        # Points for the left image
        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        # Points for the right image
        points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        # Transform points
        #points2_transformed = cv2.perspectiveTransform(points2, H)
        points2_transformed = self.perspective_transform(points2, H)
        all_points = np.concatenate((points1, points2_transformed), axis=0)

        # Find the bounding box for the new image
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # Translation matrix
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(H)
        b= (x_max - x_min, y_max - y_min)
        # Warp the left image
        #output_img = cv2.warpPerspective(left_img, H_translation, b)
        output_img = self.warp_perspective(left_img, H_translation, b)
        rows_right, cols_right = right_img.shape[:2]
        y_offset = -y_min
        x_offset = -x_min
        output_img[y_offset:y_offset + rows_right, x_offset:x_offset + cols_right] = right_img


        return output_img

    def get_keypoint(self, left_img, right_img):
        l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
        key_points2, descriptor2 = sift.detectAndCompute(r_img, None)

        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoint(self, key_points1, key_points2, descriptor1, descriptor2):
        # k-Nearest neighbours between each descriptor
        #f_mat=cv2.BFMatcher()
        #matcher = BFMatcherScratch(normType='L2', crossCheck=True)
        matcher=cv2.BFMatcher()
        matches = matcher.match(descriptor1, descriptor2)
        
        # Sort matches by distance
        #matches = sorted(matches, key=lambda x: x.distance)    opencv
        matches = sorted(matches, key=lambda x: x.distance)         

        good_matches = []
        for m in matches:
            good_matches.append(
                [key_points1[m.queryIdx].pt[0], key_points1[m.queryIdx].pt[1],
                 key_points2[m.trainIdx].pt[0], key_points2[m.trainIdx].pt[1]]
            )

        return good_matches

    def homography(self, points):
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H /= H[2, 2]
        return H

    def ransac(self, good_pts):
        best_inliers = []
        final_H = []
        t = 5
        for _ in range(10):
            random_pts = random.sample(good_pts, 4)  # Randomly select 4 points
            H = self.homography(random_pts)
            inliers = []

            for pt in good_pts:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)
                Hp /= Hp[2]
                dist = np.linalg.norm(p_1 - Hp)

                if dist < t:
                    inliers.append(pt)

            if len(inliers) > len(best_inliers):
                best_inliers, final_H = inliers, H
                
        return final_H
    

    def perspective_transform(self , points, H):

    # Convert points to homogeneous coordinates (x, y, 1)
        num_points = points.shape[0]
        points_homogeneous = np.hstack((points[:, 0, :], np.ones((num_points, 1))))  # Shape (N, 3)

    # Apply the homography matrix
        transformed_points_homogeneous = points_homogeneous @ H.T  # Shape (N, 3)

    # Convert back to Cartesian coordinates (x, y)
    # Normalize by the third coordinate (w)
        w = transformed_points_homogeneous[:, 2].reshape(-1, 1)  # Shape (N, 1)
        transformed_points = transformed_points_homogeneous[:, :2] / w  # Shape (N, 2)

    # Reshape to (N, 1, 2)
        transformed_points = transformed_points.reshape(num_points, 1, 2)
    
        return transformed_points
    

    def warp_perspective(self,img, H, output_size):
   
        width, height = output_size
        warped_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)

    # Iterate over each pixel in the output image
        for y in range(height):
            for x in range(width):
            # Create the homogeneous coordinates of the output pixel
                pixel_homogeneous = np.array([x, y, 1])
            # Invert the homography matrix to find the corresponding pixel in the input image
                input_coords = np.linalg.inv(H) @ pixel_homogeneous
                input_coords /= input_coords[2]  # Normalize

                input_x, input_y = int(input_coords[0]), int(input_coords[1])

            # Check if the calculated input coordinates are within the bounds of the input image
                if 0 <= input_x < img.shape[1] and 0 <= input_y < img.shape[0]:
                    warped_img[y, x] = img[input_y, input_x]

        return warped_img



    

