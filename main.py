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
            list_of_images.append((i, image))
    return list_of_images


def create_zone(image: np.ndarray):
    """
    Divise l'image en 9 zones et calcule la somme de chaque zone de pixels
      noirs
    :param image: L'image
    :return: Les zones
    """
    zones = []

    # image have dict values ??

    height, width = image.shape[:2]
    zone_height = height // 7
    zone_width = width // 5

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

    distance_from_image1 = calculate_distance(
        unknown_image, known_image1[1])
    distance_from_image2 = calculate_distance(
        unknown_image, known_image2[1])

    print("image 1 " + str(distance_from_image1) +
          " class :" + str(known_image1[0]))
    print("image 2 " + str(distance_from_image2) +
          " class :" + str(known_image2[0]))

    if distance_from_image1 < distance_from_image2:
        return known_image1[0]
    else:
        return known_image2[0]


def calculate_distance(unknown_image, known_image):
    """
    Calculate the distance between two images
    :param unknown_image: The unknown image (numpy array)
    :param known_image: The known image (numpy array)
    :return: The distance (float)
    """

    unknown_image_zones = create_zone(unknown_image)
    known_image_zones = create_zone(known_image)

    distance = 0

    for i in range(len(unknown_image_zones)):
        distance += (unknown_image_zones[i] - known_image_zones[i]) ** 2

    return np.sqrt(distance)


def main():
    list_of_images = add_image_to_list()

    correct_predictions = 0
    total_tests = 0

    for unknown_image_index in range(len(list_of_images)):
        unknown_image = list_of_images[unknown_image_index][1]
        true_class = list_of_images[unknown_image_index][0]

        closest_class = None
        min_distance = float('inf')

        for i in range(len(list_of_images)):
            if i != unknown_image_index:
                known_image_class = list_of_images[i][0]
                known_image = list_of_images[i][1]

                distance = calculate_distance(unknown_image, known_image)

                if distance < min_distance:
                    min_distance = distance
                    closest_class = known_image_class

        total_tests += 1
        if closest_class == true_class:
            correct_predictions += 1

    success_rate = (correct_predictions / total_tests) * 100

    print("Success rate: {:.2f}%".format(success_rate))

    # Tableau récapitulatif
    print("Classified Image | True Image")
    for i in range(len(list_of_images)):
        unknown_image_class = list_of_images[i][0]
        predicted_image_class = None
        min_distance = float('inf')

        for j in range(len(list_of_images)):
            if j != i:
                known_image_class = list_of_images[j][0]
                known_image = list_of_images[j][1]

                distance = calculate_distance(
                    list_of_images[i][1], known_image)

                if distance < min_distance:
                    min_distance = distance
                    predicted_image_class = known_image_class

        print("{:<16} | {:<11}".format(
            predicted_image_class, unknown_image_class))


main()

# Retour : dire les caractéristique choisis +image de la matrice
# et à la fin un classifieur+ résultat obtenue et dire le
# taux d'apprentissage sur une image trouvé
