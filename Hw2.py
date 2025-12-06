import numpy as np
# __________start of block__________
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            neighborhood = padded_image[i-1:i+2, j-1:j+2]
            gradient_x[i-1, j-1] = np.sum(neighborhood * sobel_x)
            gradient_y[i-1, j-1] = np.sum(neighborhood * sobel_y)
            
    return gradient_x, gradient_y


def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude


def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    gradient_direction = np.degrees(np.arctan2(sobel_y, sobel_x))
    return gradient_direction



cell_size = 7

def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.float32)
    
    # 2. Compute gradients with Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    
    for i in range(1, padded_image.shape[0]-1):
        for j in range(1, padded_image.shape[1]-1):
            patch = padded_image[i-1:i+2, j-1:j+2]
            gradient_x[i-1, j-1] = np.sum(patch * sobel_x)
            gradient_y[i-1, j-1] = np.sum(patch * sobel_y)

    # 3. Compute gradient magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.degrees(np.arctan2(gradient_y, gradient_x))

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_y = image.shape[0] // cell_height
    n_cells_x = image.shape[1] // cell_width
    
    
    bin_edges = np.linspace(-180, 180, bins + 1)
    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y_start = i * cell_height
            y_end = y_start + cell_height
            x_start = j * cell_width
            x_end = x_start + cell_width
            
            cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
            cell_direction = direction[y_start:y_end, x_start:x_end]
            
            for mag, dir_angle in zip(cell_magnitude.flatten(), cell_direction.flatten()):
                bin_idx = np.digitize(dir_angle, bin_edges) - 1
                
                if bin_idx == bins:
                    bin_idx = bins - 1
                histograms[i, j, bin_idx] += mag
           
            hist_sum = np.sum(histograms[i, j])
            if hist_sum > 0:
                histograms[i, j] /= hist_sum

    return histograms