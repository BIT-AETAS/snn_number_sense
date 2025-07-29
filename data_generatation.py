# -*- encoding: utf-8 -*-

# here put the import lib
import os
import argparse
from math import ceil, cos, sin
from matplotlib.path import Path
import numpy as np
from PIL import Image

np.random.seed(2022)

parser = argparse.ArgumentParser(description='Generate data')
parser.add_argument('--output_dir', type=str, default='./datasets/train_data',
                    help='Directory to save the generated images')
parser.add_argument('--image_dist', type=str, default='uniform',
                    choices=['uniform', 'zipf'],
                    help='Distribution of the number of images per set of parameters')
parser.add_argument('--image_num', type=int, default=1000,
                    help='Number of images to generate for each set of parameters')
parser.add_argument('--height', type=int, default=227,
                    help='Height of the generated images')
parser.add_argument('--width', type=int, default=227,
                    help='Width of the generated images')
parser.add_argument('--num_sets', type=int, default=30,
                    help='Number of different sets of parameters to generate images for')
args = parser.parse_args()

def data2img(images, number, output_dir):
    """
    Converts a list of image arrays to grayscale images and saves them as JPEG files.

    Parameters:
        images (list or np.ndarray): List or array of image data, where each element is a 2D array representing an image.
        number (int or str): Identifier to include in the saved image filenames.
        output_dir (str): Directory path where the images will be saved. Created if it does not exist.

    Each image is scaled to the 0-255 range, converted to grayscale, and saved as '{number}_{i}.jpg' in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image_array in enumerate(images):
        # image_array = images[i]
        image_array *= 255
        im = Image.fromarray(np.uint8(image_array))
        im = im.convert('L')
        im.save(os.path.join(output_dir, "{}_{}.jpg".format(number, i)))

def get_distance(point1, point2):
    """
    Calculate the squared Euclidean distance between two points.

    Args:
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        int or float: The squared distance between point1 and point2.
    """
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

def compute_distances_to_point(start, point_set):
    """
    Computes the distances from a given start point to each point in a set.

    Args:
        start: The reference point from which distances are measured.
        point_set: An iterable of points to which distances will be computed.

    Returns:
        list: A list of distances from the start point to each point in point_set.

    """
    distance_set = []
    for p in point_set:
        distance_set.append(get_distance(p, start))
    return distance_set

def inpolygon(xq, yq, xv, yv):
    """
    Determines whether query points are inside or on the boundary of a polygon.

    This function reimplements MATLAB's `inpolygon` functionality using numpy and matplotlib's Path.
    Given arrays of query points (`xq`, `yq`) and polygon vertices (`xv`, `yv`), it returns two boolean arrays:
    one indicating whether each query point is inside or on the polygon, and another indicating whether each point is exactly on the boundary.

    Args:
        xq (np.ndarray): x-coordinates of query points.
        yq (np.ndarray): y-coordinates of query points.
        xv (np.ndarray): x-coordinates of polygon vertices.
        yv (np.ndarray): y-coordinates of polygon vertices.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - in_on (np.ndarray): Boolean array, True if the point is inside or on the polygon.
            - on (np.ndarray): Boolean array, True if the point is exactly on the boundary of the polygon.
    """
    # Combine xv and yv into a vertex array
    vertices = np.vstack((xv, yv)).T
    # Define Path object
    path = Path(vertices)
    # Combine xq and yq into test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # Get a mask indicating whether test_points are strictly inside the path (boolean array)
    _in = path.contains_points(test_points)
    # Get a mask indicating whether test_points are inside or on the path
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # Get a mask indicating whether test_points are on the path
    _on = _in ^ _in_on

    return _in_on, _on

def get_standard_data(number_sets, image_num, height, width, output_dir):
    '''
    A-1. a standard set : all dots had about the same radius
    number_sets: number of dots
    image_num: number of image
    height: image height
    width: image width
    output_dir: output dir
    '''
    if isinstance(image_num, int):
        image_num = [image_num] * len(number_sets)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, now_num in enumerate(number_sets):
        images = []
        # now_num = number_sets[i]
        for idx in range(image_num[i]):
            image_data = np.zeros(shape=(height, width), dtype=int)
            total_perimeter = 0.0
            total_area = 0.0
            radius_set = []
            loc_set = []
            # add first dot
            epsil = np.random.randn()
            circle_radius = 7 + 0.7 * epsil
            circle_locx = ceil(circle_radius) + np.random.randint(0, width - 2 * ceil(circle_radius))
            circle_locy = ceil(circle_radius) + np.random.randint(0, height - 2 * ceil(circle_radius))
            radius_set.append(circle_radius)
            loc_set.append([circle_locx, circle_locy])
            total_perimeter += 2 * np.pi * circle_radius
            total_area += np.pi * (circle_radius ** 2)

            while(len(radius_set) < now_num):
                epsil = np.random.randn()
                circle_radius = 7 + 0.7 * epsil
                circle_locx = ceil(circle_radius) + np.random.randint(0, width - 2 * ceil(circle_radius))
                circle_locy = ceil(circle_radius) + np.random.randint(0, height - 2 * ceil(circle_radius))

                # new and existing dot do not coincide
                distance_set = np.array(compute_distances_to_point([circle_locx, circle_locy], loc_set))
                radius_tmp = np.ones(len(radius_set)) * circle_radius + np.array(radius_set)
                radius_tmp = radius_tmp ** 2
                no2add = distance_set[distance_set < radius_tmp]
                if len(no2add) == 0:
                    radius_set.append(circle_radius)
                    loc_set.append([circle_locx, circle_locy])
                    total_perimeter += 2 * np.pi * circle_radius
                    total_area += np.pi * (circle_radius ** 2)
                
            for x in range(height):
                for y in range(width):
                    # judge point in the circle
                    distance_set = np.array(compute_distances_to_point([x, y], loc_set))
                    radius_tmp = np.array(radius_set)
                    radius_tmp = radius_tmp ** 2
                    isblackdot = distance_set[distance_set < radius_tmp]
                    if len(isblackdot) > 0:
                        image_data[y][x] = 1
            images.append(image_data)
            with open(os.path.join(output_dir, 'info.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{idx} {now_num} {total_perimeter} {total_area}\n')
        data2img(images, now_num, output_dir)

def get_same_total_area_data(number_sets, image_num, height, width, output_dir):
    '''
    A-2. all image have same total area
    number_sets: number of dots
    image_num: number of image
    height: image height
    width: image width
    output_dir: output dir
    '''
    if isinstance(image_num, int):
        image_num = [image_num] * len(number_sets)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, now_num in enumerate(number_sets):
        images = []
        # now_num = number_sets[i]
        for idx in range(image_num[i]):
            image_data = np.zeros(shape=(height, width), dtype=int)
            total_perimeter = 0.0
            total_area = 0.0
            # generate all circle radius to make areas are same
            epsil = np.random.normal(0, 1, size=(now_num))
            circle_radius = epsil * 0.7 + 7
            area_sum = np.sum(np.pi * (circle_radius ** 2))
            scale_size = np.sqrt(area_sum / 1200)
            circle_radius = circle_radius / scale_size
            total_perimeter = np.sum(2 * np.pi * circle_radius)
            total_area = np.sum(np.pi * (circle_radius ** 2))

            avg_dist = 0
            while avg_dist < 90 or avg_dist > 100:
                loc_set = []
                rad_index = 0
                while(len(loc_set) < now_num):
                    radius = circle_radius[rad_index]
                    circle_locx = ceil(radius) + np.random.randint(0, width - 2 * ceil(radius))
                    circle_locy = ceil(radius) + np.random.randint(0, height - 2 * ceil(radius))

                    # new and existing dot do not coincide
                    if len(loc_set) > 0:
                        distance_set = np.array(compute_distances_to_point([circle_locx, circle_locy], loc_set))
                        radius_tmp = np.ones(len(loc_set)) * radius + np.array(circle_radius[0:rad_index])
                        radius_tmp = radius_tmp ** 2
                        no2add = distance_set[distance_set < radius_tmp]
                        if len(no2add) == 0:
                            loc_set.append([circle_locx, circle_locy])
                            rad_index += 1
                    else:
                        loc_set.append([circle_locx, circle_locy])
                        rad_index += 1
                if now_num > 1:
                    sum_dis = 0.0
                    for k in range(len(loc_set)):
                        distance_set = np.array(compute_distances_to_point([loc_set[k][0], loc_set[k][1]], loc_set))
                        sum_dis += np.sum(np.sqrt(distance_set)) / (now_num - 1)
                    avg_dist = sum_dis / now_num
                else:
                    avg_dist = 95
                
            for x in range(height):
                for y in range(width):
                    # judge point in the circle
                    distance_set = np.array(compute_distances_to_point([x, y], loc_set))
                    radius_tmp = circle_radius
                    radius_tmp = radius_tmp ** 2
                    isblackdot = distance_set[distance_set < radius_tmp]
                    if len(isblackdot) > 0:
                        image_data[y][x] = 1
            images.append(image_data)
            with open(os.path.join(output_dir, 'info.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{idx} {now_num} {total_perimeter} {total_area}\n')
        data2img(images, now_num, output_dir)

def get_different_shape_and_conver_hull_data(number_sets, image_num, height, width, output_dir):
    '''
    A-3. different shape + convex hull
    number_sets: number of dots
    image_num: number of image
    height: image height
    width: image width
    output_dir: output dir
    '''
    if isinstance(image_num, int):
        image_num = [image_num] * len(number_sets)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, now_num in enumerate(number_sets):
        images = []
        # now_num = number_sets[i]
        for idx in range(image_num[i]):
            #define convex hull
            theta = np.random.uniform(0, 180) * np.pi / 180
            convexhull_x = np.zeros(5)
            convexhull_y = np.zeros(5)
            for th in range(1, 6):
                convexhull_x[th-1] = round(width / 2) + cos(theta + (th - 1) * 2 * np.pi / 5) * 110
                convexhull_y[th-1] = round(height / 2) + sin(theta + (th - 1) * 2 * np.pi / 5) * 110

            image_data = np.zeros(shape=(height, width), dtype=int)
            total_perimeter = 0.0
            total_area = 0.0
            radius_set = []
            loc_set = []

            while(len(radius_set) < now_num):
                epsil = np.random.randn()
                circle_radius = 7 + 0.7 * epsil
                circle_locx = ceil(circle_radius) + np.random.randint(0, width - 2 * ceil(circle_radius))
                circle_locy = ceil(circle_radius) + np.random.randint(0, height - 2 * ceil(circle_radius))

                # new dot in convex hull
                in_convex_hull, _ = inpolygon(np.array([circle_locx]), np.array([circle_locy]), convexhull_x, convexhull_y)
                if not in_convex_hull:
                    continue

                # new and existing dot do not coincide
                distance_set = np.array(compute_distances_to_point([circle_locx, circle_locy], loc_set))
                radius_tmp = np.ones(len(radius_set)) * circle_radius + np.array(radius_set)
                radius_tmp = radius_tmp ** 2
                no2add = distance_set[distance_set < 2 * radius_tmp]
                if len(no2add) == 0:
                    radius_set.append(circle_radius)
                    loc_set.append([circle_locx, circle_locy])

            for index, radius in enumerate(radius_set):
                # radius = radius_set[index]
                loc_x = loc_set[index][0]
                loc_y = loc_set[index][1]
                randt = np.random.uniform(0, 1)

                if randt > 0.75:
                    # rectangle
                    for x in range(height):
                        for y in range(width):
                            if abs(loc_x - x) <= radius and abs(loc_y - y) <= radius:
                                image_data[y][x] = 1
                    total_perimeter += 4 * (2 * radius)
                    total_area += 4 * (radius ** 2)
                elif 0.5 < randt <= 0.75:
                    # circle
                    for x in range(height):
                        for y in range(width):
                            distance_set = np.array(compute_distances_to_point([x, y], [[loc_x, loc_y]]))
                            radius_tmp = np.array([radius])
                            radius_tmp = radius_tmp ** 2
                            isblackdot = distance_set[distance_set < radius_tmp]
                            if len(isblackdot) > 0:
                                image_data[y][x] = 1
                    total_perimeter += 2 * np.pi * radius
                    total_area += np.pi * (radius ** 2)
                elif 0.25 < randt <= 0.5:
                    # ellipse
                    for x in range(height):
                        for y in range(width):
                            tmp_x = (x - loc_x) ** 2
                            tmp_y = (y - loc_y) ** 2
                            if tmp_x / (radius ** 2) + tmp_y / ((0.5 * radius) ** 2) < 1:
                                image_data[y][x] = 1
                    # c ~ pi * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)))
                    total_perimeter += np.pi * radius * (4.5 - np.sqrt(8.75))
                    total_area += np.pi * radius * 0.5 * radius
                else:
                    # triangle
                    theta = np.random.uniform(0, 180) * np.pi / 180
                    triangle_x = np.zeros(3)
                    triangle_y = np.zeros(3)
                    for th in range(0, 3):
                        triangle_x[th] = loc_x + radius * cos(theta + th  * 2 * np.pi / 3)
                        triangle_y[th] = loc_y + radius * sin(theta + th * 2 * np.pi / 3)
                    for x in range(height):
                        for y in range(width):
                            in_convex_hull, _ = inpolygon(np.array([x]), np.array([y]), triangle_x, triangle_y)
                            if in_convex_hull:
                                image_data[y][x] = 1
                    total_perimeter = np.sqrt((triangle_x[0] - triangle_x[1]) ** 2 + (triangle_y[0] - triangle_y[1]) ** 2) + \
                                      np.sqrt((triangle_x[1] - triangle_x[2]) ** 2 + (triangle_y[1] - triangle_y[2]) ** 2) + \
                                      np.sqrt((triangle_x[2] - triangle_x[0]) ** 2 + (triangle_y[2] - triangle_y[0]) ** 2)
                    # s = 0.5 * (x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
                    total_area += 0.5 * (triangle_x[0] * triangle_y[1] + triangle_x[1] * triangle_y[2] + 
                                         triangle_x[2] * triangle_y[0] - triangle_x[1] * triangle_y[0] - 
                                         triangle_x[2] * triangle_y[1] - triangle_x[0] * triangle_y[2])
            images.append(image_data)
            with open(os.path.join(output_dir, 'info.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{idx} {now_num} {total_perimeter} {total_area}\n')
        data2img(images, now_num, output_dir)

def main():
    numbers = list(range(1, args.num_sets + 1))
    if args.image_dist == 'uniform':
        image_nums = [args.image_num] * len(numbers)
    elif args.image_dist == 'zipf':
        image_nums = np.array([i for i in range(1, 31, 1)])
        image_nums = (args.image_num / image_nums).astype(int)
    else:
        raise ValueError("Invalid image distribution type. Choose 'uniform' or 'zipf'.")
    get_standard_data(numbers, image_nums, args.height, args.width,
                      os.path.join(args.output_dir, 'standard'))
    get_same_total_area_data(numbers, image_nums, args.height, args.width,
                             os.path.join(args.output_dir, 'same_area'))
    get_different_shape_and_conver_hull_data(numbers, image_nums, args.height, args.width,
                                              os.path.join(args.output_dir, 'diff_shape'))

if __name__ == "__main__":
    main()
