import scipy
import skimage
import numpy as np
from skimage import data
from medpy.io import load
from medpy.io import save
from scipy import ndimage
from skimage import filters
from skimage import transform
import matplotlib.pyplot as plt
import scipy.interpolate as interpn
import scipy.interpolate as interpolate
from skimage.metrics import structural_similarity as ssim

#load data using medpy
image_data, image_header = load('test_trus.gipl')
# angle of rotation -15 degrees
angle = np.pi * (-900 / 180)

new_image = skimage.transform.rotate(image=image_data, angle=angle, resize=True, center=None, order=None, mode='constant', cval=0, clip=True, preserve_range=True)

"""
    The original image data was rotated by skimage.transform.rotate to obtain the rotated image array
    The angle of rotation is -15 degrees
    image_data: 3-D numpy array from the original image file
    angle: angle chosen to rotate the image anti clockwise
    interpolation mthod: constant
    
"""
# re-slicing after interpolation
# original image dimensions
D1 = image_data.shape[0]
D2 = image_data.shape[1]
D3 = image_data.shape[2]

new_x = np.linspace(-(D1-1)/2, (D1-1)/2, D1)
new_y = np.linspace(-(D2-1)/2, (D2-1)/2, D2)
new_z = np.linspace(-(D3-1)/2, (D3-1)/2, D3)
n_x, n_y, n_z = np.meshgrid(new_x, new_y, new_z, indexing = 'ij')

# rotated image dimensions
d1 = new_image.shape[0]
d2 = new_image.shape[1]
d3 = new_image.shape[2]

old_x = np.linspace(-(d1-1)/2, (d1-1)/2, d1)
old_y = np.linspace(-(d2-1)/2, (d2-1)/2, d2)
old_z = np.linspace(-(d3-1)/2, (d3-1)/2, d3)

o_x, o_y, o_z = np.meshgrid(old_x, old_y, old_z, indexing='ij')
"""
    The rotated image data was the interpolated and desired slice was extracted to obtain a non orthogonal slice
    
"""
# interpolation to desired dimensions
interpolation = interpolate.interpn ((old_x, old_y, old_z), new_image, (n_x, n_y, n_z), method = 'nearest', bounds_error = True, fill_value = 3)

# reslicing and comparing original and re-sliced images
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
ax[0].imshow(image_data[:,:,25].T, cmap=plt.cm.gray)
ax[0].set_axis_off()
ax[1].imshow(interpolation[:,:,25].T, cmap=plt.cm.gray)
ax[1].set_axis_off()
fig.savefig('original and resliced images.png', dpi=400,bbox_inches='tight')
fig.tight_layout()

# 2D bilateral filtering
from bilateral_filter_2d import bilateral_filter_2d

# edge detection
# 2D bilateral filtering with sigma_r=1, sigma_d=10 
image_bl_2d_filter_110 = bilateral_filter_2d(interpolation[:,:,25].T, 7, 1, 10)
# Plotting bilateral filter and unfiltered image
fig, ax = plt.subplots()
plt.imshow(image_bl_2d_filter_110, interpolation="bilinear", cmap='gray')
plt.axis('off')
fig.savefig('2D 1_10.png', dpi=400,bbox_inches='tight')
plt.show()

# 2D bilateral filtering with sigma_r=10, sigma_d=30 
image_bl_2d_filter_1030 = bilateral_filter_2d(interpolation[:,:,25].T, 7, 10, 30)
# Plotting bilateral filter and unfiltered image
fig, ax = plt.subplots()
plt.imshow(image_bl_2d_filter_1030, interpolation="bilinear", cmap='gray')
plt.axis('off')
fig.savefig('2D 10_30.png', dpi=400,bbox_inches='tight')
plt.show()

# 2D bilateral filtering with sigma_r=10, sigma_d=100 
image_bl_2d_filter_10100 = bilateral_filter_2d(interpolation[:,:,25].T, 7, 10, 100)
# Plotting bilateral filter and unfiltered image
fig, ax = plt.subplots()
plt.imshow(image_bl_2d_filter_10100, interpolation="bilinear", cmap='gray')
plt.axis('off')
fig.savefig('2D 10_100.png', dpi=400,bbox_inches='tight')
plt.show()


# Edge detection  with sobel filter
edge_sobel_original = filters.sobel(interpolation[:,:,20].T)
edge_sobel_filtered1 = filters.sobel(image_bl_2d_filter_110)
edge_sobel_filtered2 = filters.sobel(image_bl_2d_filter_1030)
edge_sobel_filtered3 = filters.sobel(image_bl_2d_filter_10100)

# images to visualize edge detection
fig, ax = plt.subplots()
ax.imshow(edge_sobel_original, cmap=plt.cm.gray)
ax.set_axis_off()
fig.savefig('sobel edge detection original.png', dpi=400,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.imshow(edge_sobel_filtered1, cmap=plt.cm.gray)
ax.set_axis_off()
fig.savefig('sobel edge detection 1_10.png', dpi=400,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.imshow(edge_sobel_filtered2, cmap=plt.cm.gray)
ax.set_axis_off()
fig.savefig('sobel edge detection 10_30.png', dpi=400,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.imshow(edge_sobel_filtered3, cmap=plt.cm.gray)
ax.set_axis_off()
fig.savefig('sobel edge detection 10_100.png', dpi=400,bbox_inches='tight')
plt.show()

# calculation of mean squared error, peak noise to signal ratio and structural similarity index metric
MSE1 = np.mean((interpolation[:,:,25].T- image_bl_2d_filter_110) ** 2)
print(MSE1)
PSNR1 = 20 * np.log10(255/np.sqrt(MSE1))
print(PSNR1)
SSIM1 = ssim(interpolation[:,:,25].T, image_bl_2d_filter_110)
print(SSIM1)

MSE2 = np.mean((interpolation[:,:,25].T- image_bl_2d_filter_1030) ** 2)
print(MSE2)
PSNR2 = 20 * np.log10(255/np.sqrt(MSE2))
print(PSNR2)
SSIM2 = ssim(interpolation[:,:,25].T, image_bl_2d_filter_1030)
print(SSIM2)

MSE3 = np.mean((interpolation[:,:,25].T- image_bl_2d_filter_10100) ** 2)
print(MSE3)
PSNR3 = 20 * np.log10(255/np.sqrt(MSE3))
print(PSNR3)
SSIM3 = ssim(interpolation[:,:,25].T, image_bl_2d_filter_10100)
print(SSIM3)

# 3D bilateral filtering
from bilateral_filter_3d import bilateral_filter_3d

# 3D bilateral filtering with sigma_r=1, sigma_d=10 
image_bl_3d_filter110 = bilateral_filter_3d(image = image_data, d=7, sigma_r=1, sigma_d=10)
# save 3D filtered image
save(image_bl_3d_filter110,'filtered_3d_image 1_10.gipl', image_header)

# 3D bilateral filtering with sigma_r=10, sigma_d=30 
image_bl_3d_filter1030 = bilateral_filter_3d(image = image_data, d=7, sigma_r=10, sigma_d=30)
# save 3D filtered image
save(image_bl_3d_filter1030,'filtered_3d_image 10_30.gipl', image_header)

# 3D bilateral filtering with sigma_r=10, sigma_d=100
image_bl_3d_filter10100 = bilateral_filter_3d(image = image_data, d=7, sigma_r=10, sigma_d=100)
# save 3D filtered image
save(image_bl_3d_filter10100,'filtered_3d_image 10_100.gipl', image_header)

# Edge detection  with gaussian_gradient_magnitude
# edge detection for unfiltered image data
gauss_grad_original = scipy.ndimage.gaussian_gradient_magnitude(image_data, sigma=5)
save(gauss_grad_original,'gauss_grad_original_3d_image.gipl', image_header)

# edge detection for 3D filtered image data
gauss_grad_filtered110 = scipy.ndimage.gaussian_gradient_magnitude(image_bl_3d_filter110, sigma=5)
save(gauss_grad_original,'gauss_grad_filtered_3d_image_1_10.gipl', image_header)

gauss_grad_filtered1030 = scipy.ndimage.gaussian_gradient_magnitude(image_bl_3d_filter1030, sigma=5)
save(gauss_grad_original,'gauss_grad_filtered_3d_image_10_30.gipl', image_header)

# edge detection for 3D filtered image data
gauss_grad_filtered10100 = scipy.ndimage.gaussian_gradient_magnitude(image_bl_3d_filter10100, sigma=5)
save(gauss_grad_original,'gauss_grad_filtered_3d_image_10_100.gipl', image_header)

# calculation of mean squared error, peak noise to signal ratio and structural similarity index metric
MSE1_10_3D = np.mean((image_data- image_bl_3d_filter110) ** 2)
print(MSE1_10_3D)
PSNR1_10_3D = 20 * np.log10(255/np.sqrt(MSE1_10_3D))
print(PSNR1_10_3D)
SSIM1_10_3D = ssim(image_data, image_bl_3d_filter110)
print(SSIM1_10_3D)

MSE10_30_3D = np.mean((image_data- image_bl_3d_filter1030) ** 2)
print(MSE10_30_3D)
PSNR10_30_3D = 20 * np.log10(255/np.sqrt(MSE10_30_3D))
print(PSNR10_30_3D)
SSIM10_30_3D = ssim(image_data, image_bl_3d_filter1030)
print(SSIM10_30_3D)

MSE10_100_3D = np.mean((image_data- image_bl_3d_filter10100) ** 2)
print(MSE10_100_3D)
PSNR10_100_3D = 20 * np.log10(255/np.sqrt(MSE10_100_3D))
print(PSNR10_100_3D)
SSIM10_100_3D = ssim(image_data, image_bl_3d_filter10100)
print(SSIM10_100_3D)

