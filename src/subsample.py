import numpy as np

def subsample_columns_equi(image, factor):
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


def subsample_rows_equi(image, factor):
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

def subsample_columns_random(image, factor):
    """
    Subsample input image by randomly sampling columns, reducing the number of samples by a specified factor
    Set the remainder to 0.

    Parameters:
    - image: numpy array, input image with shape (height, width)
    - factor: int, subsampling factor for columns

    Returns:
    - subsampled_image: numpy array, subsampled image
    """
    if factor <= image.shape[1] // 2:
        subsampled_image = np.zeros_like(image)
        num_samples = image.shape[1] // factor
        random_idx = np.random.permutation(image.shape[1])[:num_samples]
        subsampled_image[:, random_idx] = image[:, random_idx]
    else:
        print("error, factor chosen is too large")
        subsampled_image = image.copy()  # Return a copy to avoid modifying the original
    return subsampled_image


def subsample_rows_random(image, factor):
    """
    Subsample input image by randomly sampling rows, reducing the number of samples by a specified factor
    Set the remainder to 0.

    Parameters:
    - image: numpy array, input image with shape (height, width)
    - factor: int, subsampling factor for rows

    Returns:
    - subsampled_image: numpy array, subsampled image
    """
    if factor <= image.shape[0] // 2:
        subsampled_image = np.zeros_like(image)
        num_samples = image.shape[0] // factor
        random_idx = np.random.permutation(image.shape[0])[:num_samples]
        subsampled_image[random_idx,:] = image[random_idx,:]
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


