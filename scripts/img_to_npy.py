import cv2
import numpy as np
from .color_classes import color_classes

def img_to_npy(input_path, out_path):
    # Učitavanje slike za trening i legende
    slo_image = cv2.imread(input_path)

    # Inicijalizacija matrice za čuvanje rezultata
    out_slo = np.zeros((slo_image.shape[0], slo_image.shape[1]), dtype=np.uint8)

    # Pretraživanje slike za trening i mapiranje boja na klase
    for i in range(slo_image.shape[0]):
        for j in range(slo_image.shape[1]):
            color_temp = tuple(slo_image[i, j])
            color = set(color_temp)
            for key in color_classes.keys():
                key = set(key)
                if color == key:
                    out_slo[i, j] = map_values_to_key(color, color_classes)

    # Čuvanje rezultujuće matrice
    save(out_path, out_slo)


def save(out_path, out_npy):
    out_npy = out_npy.astype(int)
    np.save(out_path, out_npy)


def map_values_to_key(my_set, color_classes):
    for key, value in color_classes.items():
        if set(key) == my_set:
            return value
    return None