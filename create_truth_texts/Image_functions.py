''' This Python module provides utility functions for image manipulation using OpenCV. '''
import cv2
from PIL import Image
import numpy as np
import os
import imghdr
import re
import datetime
from datetime import datetime
import shutil

def prepare_img_for_ocr(image):
    """
    This function prepares an image for Optical Character Recognition (OCR) by performing several preprocessing steps.
    The input can be a path to an image file, or an image in OpenCV or PIL format.

    Parameters:
    image: str or OpenCV/PIL Image
        The input image to be prepared for OCR. This can be a path to an image file, or an image in OpenCV or PIL format.

    Returns:
    img: OpenCV Image
        The processed image ready for OCR.
    """    
    # Load the image as an OpenCV image
    img = load_image(image)   
    # Remove lines from the image
    img = remove_lines(img)    
    # Convert the image to black and white to ensure the background is white
    img, _ = convert_to_bw(img)   
    # Add a white border to the image
    img = add_border(img, 0.1)
    return img

def remove_lines(image):
    # Convert the input image to binary (if not already binary)
    _, binary_img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

    # Create structuring elements for horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Apply morphological operations to remove lines
    temp_img = cv2.erode(binary_img, horizontal_kernel, iterations=2)
    horizontal_lines = cv2.dilate(temp_img, horizontal_kernel, iterations=2)

    temp_img = cv2.erode(binary_img, vertical_kernel, iterations=2)
    vertical_lines = cv2.dilate(temp_img, vertical_kernel, iterations=2)

    # Combine horizontal and vertical lines and subtract from the original image
    combined_lines = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    img_without_lines = cv2.addWeighted(binary_img, 1, combined_lines, -1, 0)

    return img_without_lines

def add_border(image, border_percent):
    # Determine if the image is grayscale (2D) or RGB (3D)
    if len(image.shape) == 2:
        # Grayscale image
        height, width = image.shape
        new_image = np.ones((height, width), dtype=np.uint8) * 255
    elif len(image.shape) == 3:
        # RGB image
        height, width, channels = image.shape
        new_image = np.ones((height, width, channels), dtype=np.uint8) * 255
    else:
        raise ValueError("Unsupported image format")

    # Calculate border size based on the percentage of the image's width or height
    border_size = int(max(width, height) * border_percent)

    # Create a new canvas with white color
    new_width = width + 2 * border_size
    new_height = height + 2 * border_size

    if len(image.shape) == 2:
        # For grayscale images
        new_image = np.ones((new_height, new_width), dtype=np.uint8) * 255
    elif len(image.shape) == 3:
        # For RGB images
        new_image = np.ones((new_height, new_width, channels), dtype=np.uint8) * 255

    # Calculate position to paste the original image
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    # Paste the original image onto the new canvas
    if len(image.shape) == 2:
        new_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    elif len(image.shape) == 3:
        new_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = image

    return new_image

def preprocess_image(image,target_text_height=27, dpi=None):
        img = load_image(image) 
        if target_text_height is not None:
           # target_text_height = 27  # Adjust based on your requirements
            scale_factor = target_text_height / img.shape[0]
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        if dpi is not None:
             img = improve_image_dpi(img, dpi)
       # img = apply_erosion_dilation(img, operation='dilation', kernel_size=1)
        img=remove_lines(img)        
        #make sure the bg is white
        img, _ = convert_to_bw(img)
        #add white border to the img
        img = add_border(img,0.25) 
        return img

def improve_image_dpi(cv2_img, dpi):
    # Convert the cv2 image to a PIL image
    img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    # Get the size of the image in pixels
    width, height = img.size

    # Get the current DPI
    current_dpi = img.info.get('dpi', (72, 72))  # Default to 72 DPI if no DPI information is present
    #print(f"Current DPI: {current_dpi}")

    # Calculate the new size of the image based on the desired DPI
    new_size = (width * dpi // current_dpi[0], height * dpi // current_dpi[1])

    # Resize the image
    img_resized = img.resize(new_size)

    # Print the new DPI
   # print(f"New DPI: {dpi}")

    # Convert the PIL image back to a cv2 image
    cv2_img_resized = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    return cv2_img_resized

def apply_erosion_dilation(image, operation='erosion', kernel_size=1):

    # Define the kernel for erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply erosion or dilation
    if operation == 'erosion':
        processed_image = cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        processed_image = cv2.dilate(image, kernel, iterations=1)
    else:
        raise ValueError("Operation must be 'erosion' or 'dilation'")
 
    return processed_image

def convert_to_bw_(image):
        """
        Convert input image to a black and white image with inverted colors if necessary.

        Args:
            image: Input image, which can be a cv2 image, PIL image, or image file path.

        Returns:
            cv2 image: Processed black and white image with inverted colors if necessary.
        """
        # Check the type of input image and handle accordingly
        if isinstance(image, str):
            # Load the image from file path
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            # If the input is already a numpy array (cv2 image), no need to convert
            img = image
        elif isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            img = np.array(image)
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Unsupported input type. It should be cv2 image, PIL image, or image path.")

        # Threshold to convert to black and white
        _, img_bw_2d = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

        # Count how many pixels are white
        white_pixels = np.sum(img_bw_2d == 255)
    
        # Calculate the percentage of white pixels
        white_ratio = white_pixels / img.size
        #print(f"White ratio = {white_ratio}")

        # Define a threshold percentage
        threshold = 0.51

        # Check if the background is white
        if white_ratio >= threshold:
            pass
            #print('Background is white')
        else:
          #  print('Background is black')
            # Invert the image
            img_bw_2d = cv2.bitwise_not(img_bw_2d)

        # Save the image 
        # cv2.imwrite('output_x2.tiff', img_bw_2d)

        return img_bw_2d 

def convert_to_bw(image, threshold = 0.51):
    """
    Convert input image to a black and white image with inverted colors if necessary.

    Args:
        image: Input image, which can be a cv2 image, PIL image, or image file path.

    Returns:
        tuple: Processed black and white image (2D array) and color image (3D array).
    """
    # Check the type of input image and handle accordingly
    if isinstance(image, str):
        # Load the image from file path
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        # If the input is already a numpy array (cv2 image), no need to convert
        img = image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported input type. It should be cv2 image, PIL image, or image path.")

    # Apply threshold to get image with only black and white
    _, img_bw_2d = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img_bw_2d = cv2.dilate(img_bw_2d, kernel, iterations=1)
    img_bw_2d = cv2.erode(img_bw_2d, kernel, iterations=1)

    # Threshold to convert to black and white
    _, img_bw_2d = cv2.threshold(img_bw_2d, 90, 255, cv2.THRESH_BINARY)

    # Count how many pixels are white
    white_pixels = np.sum(img_bw_2d == 255)
    
    # Calculate the percentage of white pixels
    white_ratio = white_pixels / img_bw_2d.size

    # Define a threshold percentage
    

    # Check if the background is white
    if white_ratio >= threshold:
        pass
    else:
        # Invert the image
        img_bw_2d = cv2.bitwise_not(img_bw_2d)

    # Convert 2D grayscale image back to 3D for compatibility with color processing
    img_bw_3d = cv2.cvtColor(img_bw_2d, cv2.COLOR_GRAY2BGR)

    return img_bw_2d, img_bw_3d

def load_image(image):
    """
    Loads an image in various formats (path, NumPy array, or PIL Image).

    Args:
        image (str or numpy.ndarray or PIL.Image.Image): The image to be loaded.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    if isinstance(image, str):
        image_path = image
        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        image_path = None
    elif isinstance(image, Image.Image):
        image_path = None
        image = np.array(image)
    else:
        raise ValueError("Unsupported image type")
        
    return image       

def wait_for_first_image(capture):
    """
    Waits for the first captured image by continuously checking for the timestamp of the first image.

    Args:
        capture (Capture): The Capture object responsible for image capture.

    Returns:
        str: The timestamp of the first captured image.
    """
    while capture.first_img_ts is None:
        pass
    #print('first_img_ts:   ', capture.first_img_ts)
    return capture.first_img_ts
   
def find_next_image(capture, first_img_ts,extract_parameters_from_img):
    """
    Finds the next image captured by the Capture device after a given timestamp and processes it.

    Args:
        capture (Capture): The Capture object responsible for image capture.
        first_img_ts (str): The timestamp of the first captured image.
        extract_parameters_from_img (function): A function to extract parameters from the found image.

    Returns:
        None
    """
    my_folders = MyFolders()
    img_ts = dt.datetime.strptime(first_img_ts, '%Y-%m-%d %H:%M:%S')
    
    while True:
        current_img_ts = dt.datetime.strptime(capture.current_img_ts, '%Y-%m-%d %H:%M:%S')
        
        if img_ts <= current_img_ts:
            img_path = my_folders.get_image_path(capture, img_ts)
            
            if os.path.exists(img_path):
                #print(img_path, '>>>>>>>>>i found one image>>>>>>>>>> ')
                extract_parameters_from_img(current_img_ts, img_path)  # process the found image
                
            img_ts += dt.timedelta(seconds=1)
            sleep(0.1)

def show_image(img):

    # Check if the image was successfully loaded
    if img is not None:
        # Display the image in a window
        cv2.imshow('Image', img)

        # Wait for a key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image not found.")
 
def resize_image_cv2(input_image, new_size):  
    """
    Resizes an input image using OpenCV.
         input_image (numpy.ndarray): The original image to be resized.
        new_size (dict): A dictionary containing the target width and height.
            Example: {"width": 640, "height": 480}

    Returns:
        numpy.ndarray or None: The resized image if successful, or None if an error occurs.
    """
    try:
        # Check the original image size
       #print("Original image shape:", input_image.shape)  # Add this line
        original_height, original_width, _ = input_image.shape

        # Extract width and height from the new_size dictionary
        target_width = new_size["width"]
        target_height = new_size["height"]

        # Check if resizing is necessary
        if (original_width, original_height) == (target_width, target_height):
            #print("Image is already of the desired size. No resizing needed.")
            return input_image

        # Resize the image
        resized_image = cv2.resize(input_image, (target_width, target_height))

        #print("Image resized successfully.")
        return resized_image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def convert_to_grayscale(image):
    """
    Converts an input color image to grayscale using OpenCV.

    Args:
        image (numpy.ndarray): The original color image (BGR format).

    Returns:
        numpy.ndarray or None: The grayscale image if successful, or None if an error occurs.
    """
    try:
 
        # Check if the image is already in grayscale
        if len(image.shape) == 2:
           # print("Image is already in grayscale.")
            return image

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

       # print("Image successfully converted to grayscale.")
        return gray_image

    except Exception as e:
        print(f"Error: {e}")
        return None
  
def percentage_white_pixels(image):
    # In a binary image, white pixels have a value of 255
    white = 255
    # Create a mask for white pixels
    white_pixels = (image == white)
    # Count the white pixels
    count_white = np.sum(white_pixels)
    # Calculate the total pixels in the image
    total_pixels = image.size
    # Calculate the percentage of white pixels
    percentage = (count_white / total_pixels) * 100
    return percentage

def percentage_of_black_pixels(img):
    """
    Calculate the percentage of black pixels in a grayscale image.

    Parameters:
    - img (ndarray): Input grayscale image.

    Returns:
    - percentage (float): Percentage of black pixels.
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image")

    # Calculate the total number of pixels
    _total_pixels = img.shape[0] * img.shape[1]
   # print('_total_pixels', _total_pixels)

    # Count the number of black pixels (pixels with intensity value 0)
    _black_pixels = np.count_nonzero(img == 0)
    #print('_black_pixels', _black_pixels)

    # Calculate the percentage of black pixels
    percentage_ = (_black_pixels / _total_pixels) * 100.0
    #print(f"Percentage of black pixels: {percentage_:.2f}%")
    return percentage_

def is_no_signal(image):
    """
    Determine if an image indicates a no signal state.

    Parameters:
    - image (ndarray): Input grayscale image.

    Returns:
    - is_no_signal (bool): True if no signal, False otherwise.
    """
    threshold=12.50
    percentage_black = percentage_of_black_pixels(image)
    return percentage_black == threshold
 
def crop_image(image, x1, x2, y1, y2):
    # Crop the image based on position coordinates
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image

def save_image_cv(image, folder_path, file_name):
    """
    Save an image to a specified folder using OpenCV.

    Args:
    image (numpy.ndarray): The image to be saved.
    folder_path (str): The path to the folder where the image will be saved.
    file_name (str): The name of the file (including extension) to be saved.

    Returns:
    None
    """
    # Ensure that the folder exists, create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Construct the full path to the image file
    file_path = os.path.join(folder_path, file_name)

    # Save the image
    try:
        cv2.imwrite(file_path, image)
        print(f"Image saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        
def is_nested_directory_empty(directory_path):
    for dirpath, dirnames, filenames in os.walk(directory_path):
        if dirnames or filenames:
            return False
    return True

def does_nested_directory_have_images(directory_path):
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if imghdr.what(os.path.join(dirpath, filename)):
                return True
    return False

def extract_timestamp_from_pattern(pattern, input_string):
    """
    This function extracts a timestamp from a string based on a given pattern.
    
    Parameters:
    pattern (str): The pattern to match in the input string. The pattern should be in the form of a regular expression.
    input_string (str): The string from which to extract the timestamp.
    
    Returns:
    str: The extracted timestamp in the format '%Y-%m-%d %H:%M:%S', or None if the input_string doesn't match the pattern.
    """
    try:
        # Use regular expression to extract the timestamp part from the input string
        match = re.search(pattern, input_string)
        if match:
            timestamp_str = match.group(1)
            
            # Parse the timestamp string into a datetime object
            extracted_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Format the extracted datetime object into the desired timestamp format
            timestamp = extracted_time.strftime('%Y-%m-%d %H:%M:%S')  # Change format as needed
            
            return timestamp
        else:
            print(f"Failed to extract timestamp. The input string '{input_string}' doesn't match the pattern '{pattern}'.")
            return None
    except ValueError:
        print(f"Failed to parse timestamp. The timestamp part '{timestamp_str}' doesn't match the format '%Y%m%d_%H%M%S'.")
        return None



def extract_ts_from_img_name(filename):
    img_name_pattern="ID\\d{4}_MID\\d{4}_(\\d{8}_\\d{6})" ## it must be loaded from json
    file_name_without_extension = os.path.splitext(filename)[0]  # Split the file name and extension
    
    timestamp = extract_timestamp_from_pattern(img_name_pattern, file_name_without_extension)
    if timestamp:
        #print("Extracted Timestamp:", timestamp)
        return timestamp
    else:
        print("Failed to extract timestamp.")
        return -1
   
 
def draw_rectangles_and_labels(image_path, par_names_and_positions_dic, par_names_and_values_dict, rectangle_color=(0 , 0, 0), text_color=(0, 0, 0), font_scale=1, font_thickness=1,  text_position=(1, 0), label_bg_color=(0, 255, 0)):
    # Load your image
    image = cv2.imread(image_path)
    
    # Set the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Iterate over your dictionary
    print(f"par_names_and_positions_dic:{par_names_and_positions_dic}")
    print(f"par_names_and_values_dict: {par_names_and_values_dict}")
    for par_name in par_names_and_positions_dic:
        print(f"par_name: {par_name}")
        # Get coordinates
        x1, y1 = int(par_names_and_positions_dic[par_name]['x1']), int(par_names_and_positions_dic[par_name]['y1'])
        x2, y2 = int(par_names_and_positions_dic[par_name]['x2']), int(par_names_and_positions_dic[par_name]['y2'])

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), rectangle_color, 2)

        # Write text
        lable_text=par_names_and_values_dict[par_name]

        # Get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(lable_text, font, font_scale, font_thickness)[0]

        # Calculate new font scale based on height of rectangle
        new_font_scale = (y2 - y1) / text_height

        # set the text start position
        text_start_x = x1 + text_position[0]
        text_start_y = y1 - text_position[1]

        # Draw a filled rectangle (as background of the text)
        cv2.rectangle(image, (text_start_x, text_start_y ), (x2, y1-(y2-y1)), label_bg_color, cv2.FILLED)

        # Then put the text on top of it
        cv2.putText(image, lable_text, (text_start_x, text_start_y), font, new_font_scale, text_color, font_thickness)

    return image





def get_image_files_in_directory(directory_path):
    """
    This function takes a directory path as input and returns a list.
    Each element in the list is a tuple containing the directory path and the image name.

    Parameters:
    directory_path (str): The path of the directory to be scanned for image files.

    Returns:
    list: A list where each element is a tuple (directory path, image name).
    
    usage example:
        for img_dir, img_name in images_list:
           print(img_dir)
           print(img_name)
    """
    
    # Initialize an empty list to store the results
    images_list = []

    # Use os.walk to iterate over all directories and files in the given directory
    for dirpath, dirnames, filenames in os.walk(directory_path):
        
        # Iterate over all files in the current directory
        for filename in filenames:
            
            # Use imghdr to check if the current file is an image file
            if imghdr.what(os.path.join(dirpath, filename)):
                
                # If the file is an image file, add it to the list
                images_list.append((dirpath, filename))  

    # Return the list containing all image files in the directory
    return images_list


def move_specific_image(src_folder, dest_folder, image_name):
    """
    This function moves a specific image file from the source folder to the destination folder.

    Parameters:
    src_folder (str): The source folder path where the image is currently located.
    dest_folder (str): The destination folder path where the image will be moved.
    image_name (str): The name of the image file to be moved.

    Returns:
    None
    """

    # Check if source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist.")
        return

    # Check if destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Construct full file path
    src = os.path.join(src_folder, image_name)
    dest = os.path.join(dest_folder, image_name)

    # Check if the file is an image
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Check if the file exists in the source folder
        if os.path.exists(src):
            # Move the file
            shutil.move(src, dest)
          #  print(f"The image file {image_name} has been moved from {src_folder} to {dest_folder}.")
        else:
            print(f"The image file {image_name} does not exist in the source folder {src_folder}.")
    else:
        print(f"The file {image_name} is not an image file.")


def crop_and_rename_and_save_image_copy(src_folder, dest_folder, image_name, new_name_suffix, x1, y1, x2, y2):
    """
    This function makes a copy of a specific image file from the source folder, crops the copy, renames it, and saves it in the destination folder using OpenCV.

    Parameters:
    src_folder (str): The source folder path where the image is currently located.
    dest_folder (str): The destination folder path where the image will be saved.
    image_name (str): The name of the image file to be copied and cropped.
    new_name_suffix (str): The suffix to be added to the original image name.
    x1, y1, x2, y2 (int): The coordinates of a rectangle for the cropping area.

    Returns:
    None
    """

    # Check if source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist.")
        return
        # Remove extension from file name if it exists
    
    # Check if destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Construct full file path
    src = os.path.join(src_folder, image_name)

    # Check if the file is an image
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Check if the file exists in the source folder
        if os.path.exists(src):
            # Read the image file
            img = cv2.imread(src)
            # Make a copy of the image
            img_copy = np.copy(img)
            # Crop the copied image
            img_cropped = img_copy[int(y1):int(y2), int(x1):int(x2)]
            # Construct new image name
            base_name, ext = os.path.splitext(image_name)
            # set the new img extention to .tif (It is required for Tesseract training)
            ext='.tif'
            new_image_name = f"{base_name}_{new_name_suffix}{ext}"
            # Construct destination file path
            dest = os.path.join(dest_folder, new_image_name)
            # Save the cropped image
            filterd_img = prepare_img_for_ocr(img_cropped)
            cv2.imwrite(dest, filterd_img)
           # print(f"A copy of the image file {image_name} has been cropped and saved as {new_image_name} in {dest_folder}.")
            # Remove extension from file name if it exists
            new_image_name, _ = os.path.splitext(new_image_name)
            return  new_image_name
        else:
            print(f"The image file {image_name} does not exist in the source folder {src_folder}.")
           
    else:
        print(f"The file {image_name} is not an image file.")
        
 

def write_truth_text(content, dir_name, file_name):
    """
    This function creates a text file with the given content in the specified directory.

    Parameters:
    content (str): The content to be written to the file.
    dir_name (str): The directory where the file will be created.
    file_name (str): The name of the file without extension.

    Returns:
    None
    """

    # Check if directory exists, if not, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Construct full file path with .gt.txt extension
    file_path = os.path.join(dir_name, f"{file_name}.gt.txt")

    # Write content to the file
    with open(file_path, 'w') as f:
        f.write(content)

   # print(f"Content has been written to {file_path}.")

def apply_filter_and_save(image_path, filter_function):
    """
    This function applies a given filter function to an image and saves the filtered image.

    Parameters:
    image_path (str): The path to the image file.
    filter_function (function): The filter function to apply to the image.

    Returns:
    None
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Apply the filter function to the image
    filtered_img = filter_function(img)

    # Save the filtered image with the same name, overwriting the original image
    cv2.imwrite(image_path, filtered_img)

def process_images_in_folder(folder_path, filter_function):
    """
    This function applies a given filter function to all .tif or .tiff images in a specified folder.

    Parameters:
    folder_path (str): The path to the folder containing the images.
    filter_function (function): The filter function to apply to the images.

    Returns:
    None
    """
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .tif or .tiff image
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Construct the full image path
            image_path = os.path.join(folder_path, filename)
            # Apply the filter and save the image
            apply_filter_and_save(image_path, filter_function)
             
def rename_tiff_to_tif(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .tiff extension
        if filename.lower().endswith('.tiff'):
            # Create the new filename by replacing .tiff with .tif
            new_filename = filename[:-5] + '.tif'
            # Get the full paths to the original and new filenames
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(original_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')
if __name__ == "__main__":
    img=load_image(r"C:\Users\nizar\OneDrive\Desktop\tst\ID0001_MID0004_20240228_131422.tiff")
    bw_img, _=convert_to_bw(img)
    save_image_cv(bw_img, r"D:\future link\AljalaliAli\train_tessarct\imges", "ID0001_MID0004_20240228_131422.tiff")