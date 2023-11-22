import numpy as np

def subsample_columns(image, factor):
    """
    Subsample input image by taking every factor'th column.
    Set the remainder to 0.

    Parameters:
    - image: numpy array, input image with shape (height, width)
    - factor: int, subsampling factor for columns

    Returns:
    - subsampled_image: numpy array, subsampled image
    """
    if factor <= image.shape[1] // 2:
        subsampled_image = np.zeros_like(image)
        subsampled_image[:, ::factor] = image[:, ::factor]
    else:
        print("error, factor chosen is too large")
        subsampled_image = image.copy()  # Return a copy to avoid modifying the original
    return subsampled_image


def subsample_rows(image, factor):
    """
    Subsample the input image by taking every factor'th row.
    Set the remainder to 0.

    Parameters:
    - image: numpy array, input image with shape (height, width)
    - factor: int, subsampling factor for rows

    Returns:
    - subsampled_image: numpy array, subsampled image
    """
    if factor <= image.shape[0] // 2:
        subsampled_image = np.zeros_like(image)
        subsampled_image[::factor, :] = image[::factor, :]
    else:
        print("error, factor chosen is too large")
        subsampled_image = image.copy()  # Return a copy to avoid modifying the original
    return subsampled_image

def subsample_center_ring(image, inner_radius, outer_radius):
    """
    Subsample input image by selecting pixels within a circular ring at the center.
    Parameters:
    - image: numpy array, input image with shape (height, width)
    - inner_radius: int, inner radius of the circular ring
    - outer_radius: int, outer radius of the circular ring
    Returns:
    - subsampled_image: numpy array, subsampled image
    """
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= outer_radius**2
    mask = mask & ((x - center_x)**2 + (y - center_y)**2 >= inner_radius**2)
    subsampled_image = np.zeros_like(image)
    subsampled_image[mask] = image[mask]
    return subsampled_image


