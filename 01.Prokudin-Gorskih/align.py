import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    high, width = raw_img.shape
    chnl_high = high // 3
    channel_bounds = [(0, chnl_high), (chnl_high, 2 * chnl_high), (2 * chnl_high, 3 * chnl_high)]
    
    channels = [raw_img[start:end, :] for start, end in channel_bounds]
    coords = [np.array([start, 0]) for start, _ in channel_bounds]

    if crop:
        crop_high, crop_width = int(chnl_high * 0.1), int(width * 0.1)
        
        for i in range(len(channels)):
            channels[i] = channels[i][crop_high:-crop_high, crop_width:-crop_width]
            coords[i] += np.array([crop_high, crop_width])

    unaligned_rgb = (channels[2], channels[1], channels[0])
    coords = (coords[2], coords[1], coords[0])

    return unaligned_rgb, coords


def mse_score(img1, img2):
    return np.mean((img1 - img2) ** 2)


def ncc_scrore(img1, img2):
    return np.sum(img1 * img2) / (np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2)))


def shift_image(img, dy, dx):
    high, width = img.shape
    y_start = max(0, dy)
    y_end = min(high, high + dy)
    x_start = max(0, dx)
    x_end = min(width, width + dx)

    return img[y_start:y_end, x_start:x_end] 


def find_relative_shift_pyramid(img_a, img_b):
    best_mse = np.inf
    
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            shifted_img_a = shift_image(img_a, -dy, -dx)
            shifted_img_b = shift_image(img_b, dy, dx)
            
            mse = mse_score(shifted_img_a, shifted_img_b)
            if mse < best_mse:
                best_mse = mse
                a_to_b = np.array([dy, dx])
    
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    red, green, blue = crops
    red_coord, green_coord, blue_coord = map(np.array, crop_coords)
    
    r_to_g_relative = find_relative_shift_fn(red, green)
    b_to_g_relative = find_relative_shift_fn(blue, green)

    r_to_g = (green_coord - red_coord) + r_to_g_relative
    b_to_g = (green_coord - blue_coord) + b_to_g_relative

    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    red, _, _ = channels
    coords = [np.array(coord) for coord in channel_coords]
    shifted_coords = [coords[0] + r_to_g, coords[1], coords[2] + b_to_g]

    y_min = max(coord[0] for coord in shifted_coords)
    y_max = min(coord[0] + channel.shape[0] for coord, channel in zip(shifted_coords, channels))
    x_min = max(coord[1] for coord in shifted_coords)
    x_max = min(coord[1] + channel.shape[1] for coord, channel in zip(shifted_coords, channels))
    
    high, width = y_max - y_min, x_max - x_min
    aligned_image = np.zeros((high, width, 3), dtype=red.dtype)

    for i, (channel, coord) in enumerate(zip(channels, shifted_coords)):
        y_start = y_min - coord[0]
        y_end = y_max - coord[0]
        x_start = x_min - coord[1]
        x_end = x_max - coord[1]
        
        aligned_image[:, :, i] = channel[y_start:y_end, x_start:x_end]

    return aligned_image


def shift_normalization(shift_y, shift_x, img_high, img_width):
    if np.abs(shift_y) > img_high / 2:
        shift_y -= img_high * np.sign(shift_y)
    if np.abs(shift_x) > img_width / 2:
        shift_x -= img_width * np.sign(shift_x)
    return shift_y, shift_x


def find_relative_shift_fourier(img_a, img_b):
    img_a_fft = np.fft.fft2(img_a)
    img_b_fft = np.fft.fft2(img_b)
    cross_correlation = np.fft.ifft2(np.conj(img_a_fft) * img_b_fft)

    max_y, max_x = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    shift_y, shift_x = shift_normalization(max_y, max_x, img_b.shape[0], img_b.shape[1])

    a_to_b = np.array([shift_y, shift_x])
    return a_to_b


if __name__ == "__main__":
    import homework_1.common as common
    import homework_1.pipeline as pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
