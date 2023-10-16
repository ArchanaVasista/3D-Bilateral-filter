import numpy as np

""" algorithm to implement 3D bilateral filter
"""

def gaussian(x,sigma):
    return (1.0/(2*np.pi)*sigma)*np.exp(-(x**2)/(2*(sigma**2)))
"""
gaussian represents a 3 dimensional guassian kernel

INPUTS:

x: X-coordinates, Y-coordinates and Z-coordinates of the input image array
sigma: standard deviation

OUTPUTS:
    guassian of the image in 3 dimension

"""
def distance_3d(x1, y1, z1, x2, y2, z2):
    return np.sqrt(np.abs((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
"""
distance_3d represents Euclidean distance between 2 coordinate points in 3D space

INPUTS:

x1, y1, z1, x2, y2, z2: X-coordinates, Y-coordinates and Z-coordinates at of the input image array
at 2 different point

OUTPUTS:
    Euclidean distance between coordinates

"""
# 3D bilateral filter
def bilateral_filter_3d(image, d, sigma_r, sigma_d):
    filtered_3d_image = np.zeros(image.shape)
    """
     a 3 dimensional bilateral filter  

    INPUTS:
    x1, y1, z1, x2, y2, z2: X-coordinates, Y-coordinates and Z-coordinates of the input image array
    d: diameter of each pixel neighborhood used for filtering. Default value is 7
    sigma_r: standard deviation(range parameter) which filters a pixel
    sigma_d: standard deviation (spatial or domain parameter) which controls downweight of adjacent pixel

    OUTPUTS:
        bilaterally filtered 3d image array
    
    """
# x, y, z coordinates of the image    
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
# normalization parameter and filtered image 
                weight_total = 0
                filtered_image = 0
                for i in range(d):
                    for j in range(d):
                        for k in range(d):
                            n_x =x - (d/2 - i)
                            n_y =y - (d/2 - j)
                            n_z =z - (d/2 - k)
                            if n_x >= 455:
                                n_x -= 455
                            if n_y >= 325:
                                n_y -= 325
                            if n_z >= 46:
                                n_z -= 46
# range Gaussian 
                            gi = gaussian(image[int(n_x)][int(n_y)][int(n_z)] - image[x][y][z], sigma_r)
# Spatial Gaussian weighting
                            gs = gaussian(distance_3d(n_x, n_y, n_z, x, y, z), sigma_d)
# range and spatial Gaussian weighting
                            weight = gi * gs
                            filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)][int(n_z)] * weight)
                            weight_total = weight_total + weight
                filtered_image = filtered_image // weight_total
                filtered_3d_image[x][y][z] = int(np.round(filtered_image))
    return filtered_3d_image
