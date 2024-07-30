"""
Requirements: db, images, config file, detect_pattern.py, Image_functions.py
"""

import os
import cv2
import sqlite3
import json
import numpy as np
import gui
from detect_pattern import *
import shutil

def read_json_file(file_path):
    """
    Read and return the content of a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def get_parameters(file_path, image_id):
    """
    Get parameters for a specific image ID from a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        image_id (str): Image ID.

    Returns:
        dict: Dictionary with parameter names and positions.
    """
    try:
        json_data = read_json_file(file_path)
        parameters = json_data['images'][str(image_id)]['parameters']
        return {value['name'].lower(): value['position'] for key, value in parameters.items()}
    except:
        return {}

def get_data_based_on_ts(db_dir, ts):
    """
    Get data from the SQLite database based on a timestamp.

    Args:
        db_dir (str): Path to the SQLite database file.
        ts (str): Timestamp to query.

    Returns:
        dict: Dictionary with column names and values.
    """
    conn = sqlite3.connect(db_dir)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    for table in tables:
        table = table[0]
        cur.execute(f"SELECT * FROM {table} WHERE ts = ?;", (ts,))
        rows = cur.fetchall()
        for row in rows:
            col_names = [description[0] for description in cur.description]
            row_dict = {col_name: col_value for col_name, col_value in zip(col_names, row)}
    conn.close()
    return row_dict

def move_specific_image(src_folder, dest_folder, image_name):
    """
    Move a specific image file from source folder to destination folder.

    Args:
        src_folder (str): Source folder path.
        dest_folder (str): Destination folder path.
        image_name (str): Image filename.

    Returns:
        None
    """
    src = os.path.join(src_folder, image_name)
    dest = os.path.join(dest_folder, image_name)

    if not os.path.exists(src):
        print(f"The image file {image_name} does not exist in the source folder {src_folder}.")
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    shutil.move(src, dest)
    print(f"Moved image {image_name} to {dest_folder}.")

def main():
    """
    Main function to read configuration, process images, and display them using a GUI.
    """
    import configparser

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    configFiles_dir = config.get('Paths', 'configFiles_dir')
    mde_config_file_name = config.get('Paths', 'mde_config_file_name')
    templates_dir_name = config.get('Paths', 'templates_dir_name')
    img_dir = config.get('Paths', 'img_dir')
    db_dir = config.get('Paths', 'db_dir')
    labeled_imgs = config.get('Parametrs', 'labeled_imgs')
    print(f'labeled_imgs: {labeled_imgs}')

    # Initialize the ImageDisplay outside the loop
    display = gui.ImageDisplay()

    images_list = get_image_files_in_directory(img_dir)

    for img_dir, img_name in images_list:
        ts = extract_ts_from_img_name(img_name)
        try:
            par_names_and_values_dict = get_data_based_on_ts(db_dir, ts)
            par_names_and_values_dict['ts']
        except:
            # If something went wrong, save the current image under "Others"
            move_specific_image(img_dir, 'Others', img_name)
            continue

        # Initialize the ImageMatcher
        matcher = ImageMatcher(configFiles_dir, mde_config_file_name, templates_dir_name)
        # Perform image matching
        match_values, temp_img_id = matcher.match_images(cv2.imread(os.path.join(img_dir, img_name)))

        if int(temp_img_id) > 0:
            par_names_and_positions_dic = get_parameters(os.path.join(configFiles_dir, mde_config_file_name), temp_img_id)
            print(f"par_names_and_positions_dic: {par_names_and_positions_dic}")

            if int(labeled_imgs) == 1:
                labeled_image = draw_rectangles_and_labels(os.path.join(img_dir, img_name), par_names_and_positions_dic, par_names_and_values_dict)
                img = labeled_image
            else:
                img = cv2.imread(os.path.join(img_dir, img_name))

            # Remove the 'ts' item from the dictionary
            del par_names_and_values_dict['ts']

            # Update the GUI with the new image
            updated_par_names_and_values_dict = gui.show(display, img, par_names_and_values_dict, 'ts')

            # Ensure GUI updates and wait for user input
            display.root.update_idletasks()
            display.root.update()

            # Check if the config issue button was clicked
            if display.config_issue_button_clicked:
                print(f"Configuration Issue button clicked for image {img_name}")
                move_specific_image(img_dir, 'Configuration Issue', img_name)
                display.config_issue_button_clicked = False  # Reset the flag
                continue  # Move to the next image

            # Compare the dictionaries to find updated parameters
            list_of_name_of_updated_parametrs = [k for k in updated_par_names_and_values_dict if updated_par_names_and_values_dict[k] != par_names_and_values_dict.get(k)]
            corrected_par_names_and_values_dict = {key: updated_par_names_and_values_dict[key] for key in list_of_name_of_updated_parametrs if key in updated_par_names_and_values_dict}
            corrected_par_names_and_positions_dic = {key: par_names_and_positions_dic[key] for key in par_names_and_positions_dic if key in corrected_par_names_and_values_dict}

            for par_name, par_pos in corrected_par_names_and_positions_dic.items():
                ground_truth_dir = 'mde-ground-truth'
                new_image_name = crop_and_rename_and_save_image_copy(img_dir, ground_truth_dir, img_name, par_name, par_pos['x1'], par_pos['y1'], par_pos['x2'], par_pos['y2'])
                write_truth_text(corrected_par_names_and_values_dict[par_name], ground_truth_dir, new_image_name)

            if corrected_par_names_and_values_dict == {}:
                move_specific_image(img_dir, r'Images_Good OCR', img_name)
            else:
                move_specific_image(img_dir, r'Images_Bad OCR', img_name)

        else:
            print("No matching template found.")
            gui.show(display, cv2.imread(os.path.join(img_dir, img_name)), par_names_and_values_dict, 'ts')
            move_specific_image(img_dir, 'No_Match_imgs', img_name)

        # Ensure GUI updates and wait for user input
        display.root.update_idletasks()
        display.root.update()

        # Check if the config issue button was clicked after processing the image
        if display.config_issue_button_clicked:
            print(f"Configuration Issue button clicked for image {img_name}")
            move_specific_image(img_dir, 'Configuration Issue', img_name)
            display.config_issue_button_clicked = False  # Reset the flag

    # Close the GUI after the loop ends
    display.root.destroy()

if __name__ == "__main__":
    main()
