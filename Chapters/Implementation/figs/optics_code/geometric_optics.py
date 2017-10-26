def image_distance(ABCD):
    'Get image distance for optical system with ray matrix `ABCD`.'
    B = ABCD[0, 1]
    D = ABCD[1, 1]
    return -B / D
