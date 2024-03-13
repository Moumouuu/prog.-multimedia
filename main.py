import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def convert_grayscale_to_black_and_white(image, threshold=0.5):
    """
    Convert the grayscale image to black and white image
    :param image: The grayscale image
    :param threshold: The threshold value
    :return: The black and white image
    """
    return np.where(image > threshold, 1.0, 0.0)


def resize_image(image):
    """
    Resize the image to new size
    :param image: The image
    :param new_size: The new size
    :return: The resized image
    """
    return cv2.resize(image, dsize=(21, 21), interpolation=cv2.INTER_CUBIC)


def show_image(image):
    """
    Show the image
    :param image: The image
    :return: None
    """
    plt.imshow(image, cmap='gray')
    plt.show()


def read_image(path):
    """
    Read the image from the given path
    :param path: The path of the image
    :return: The image
    """
    return mpimg.imread(path)


def add_image_to_list():
    """
    Add the image to the list of images
    :param image: The image
    :param list_of_images: The list of images
    :return: None
    """
    list_of_images = []

    for i in range(10):
        for j in range(1, 11):
            image = read_image('base/' + str(i) + '/' +
                               str(i) + '_' + str(j) + '.png')
            image = resize_image(image)
            image = convert_grayscale_to_black_and_white(image)

            # show_image(image)
            list_of_images.append({i: image})
    return list_of_images


def create_zone(image):
    """
    Divise l'image en 9 zones et calcule la somme de chaque zone de pixels
      noirs
    :param image: L'image
    :return: Les zones
    """
    zones = []

    height, width = image.shape[:2]
    zone_height = height // 3
    zone_width = width // 3

    for i in range(0, height, zone_height):
        for j in range(0, width, zone_width):

            # Extraire la zone
            zone_pixels = image[i:i+zone_height, j:j+zone_width, 0]

            # Compte le nombre de pixels noirs
            black_pixel_count = np.sum(zone_pixels == 0)
            zones.append(black_pixel_count)

    return zones


def compare_images(unknown_image, known_image1: dict, known_image2: dict):
    """
    compare two images and return the class of the closest image
    :param unknown_image: The unknown image
    :param known_image1: The first known image
    :param known_image2: The second known image
    :return: The class of the closest image
    """

    distance_from_image1 = calculate_distance(unknown_image, known_image1[0])
    distance_from_image2 = calculate_distance(unknown_image, known_image2[0])

    print("image 1 " + str(distance_from_image1))
    print("image 2 " + str(distance_from_image2))

    if distance_from_image1 < distance_from_image2:
        return known_image1.keys()
    else:
        return known_image2.keys()


def calculate_distance(unknown_image, known_image):
    """
    Calculate the distance between two images
    :param unknown_image: The unknown image
    :param known_image: The known image
    :return: The distance (float)
    """
    unknown_image_zones = create_zone(unknown_image)
    known_image_zones = create_zone(known_image)

    distance = 0

    for i in range(len(unknown_image_zones)):
        distance += (unknown_image_zones[i] - known_image_zones[i]) ** 2

    return np.sqrt(distance)


def main():
    # todo : bug avec les index des list of image
    list_of_images = add_image_to_list()

    print(len(list_of_images))

    print(list_of_images[1])
    print(list_of_images[12])

    print(compare_images(list_of_images[0][0],
          list_of_images[1], list_of_images[12]))


main()
