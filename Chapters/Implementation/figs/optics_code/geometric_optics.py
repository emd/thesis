class Ray(object):
    def __init__(self, rho, theta):
        '''Create an instance of `Ray` class.

        Input parameters:
        -----------------
        rho - float
            Transverse distance of beam from optical axis.
            [rho] = arbitrary units

        theta - float
            Angle between ray and optical axis.
            [theta] = rad

        '''
        self.rho = rho
        self.theta = theta

    def applyABCD(self, ABCD):
        'Apply `ABCD` ray-matrix transformation to ray.'
        A = ABCD[0, 0]
        B = ABCD[0, 1]
        C = ABCD[1, 0]
        D = ABCD[1, 1]

        rho = (A * self.rho) + (B * self.theta)
        theta = (C * self.rho) + (D * self.theta)

        return Ray(rho, theta)


def image_distance(ABCD):
    'Get image distance for optical system with ray matrix `ABCD`.'
    B = ABCD[0, 1]
    D = ABCD[1, 1]
    return -B / D
