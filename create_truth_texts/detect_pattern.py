import json
import os
from Image_functions import *



class ImageMatcher:
    """
    Class for matching input images with templates and determining the best match.
    """
    def __init__(self, configFiles_dir, mde_config_file_name, templates_dir_name):     
        # Ensure the MDE config directory exists
        if not os.path.exists(configFiles_dir):
            os.makedirs(configFiles_dir)

        # Construct the path to the MDE config file
        self.mde_config_file_path = os.path.join(configFiles_dir, mde_config_file_name)

        # Load MDE config data if the config file exists
        if os.path.exists(self.mde_config_file_path):
            self.mde_config_data = self.load_mde_config_data(self.mde_config_file_path)
        else:
            self.mde_config_data = {}  # Handle case where config file doesn't exist
        # Ensure the templates directory exists within the MDE config directory
        self.templates_dir =os.path.join(configFiles_dir, templates_dir_name)
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)

       
     
    def load_mde_config_data(self, json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file: {json_file_path}")
            print(e)
            return {}

 
    def match_images(self, img, min_match_val=0.9): 
        """
        Match input image against templates specified in the MDE configuration data.

        Parameters:
        - img (ndarray): Input image to be matched against templates.
        - min_match_val (float): Minimum match value threshold for considering a match.

        Returns:
        - Tuple: (match_values, temp_img_id) if a matching template is found,
                 (-1, -1) if no matching template is found.
        """

       #loop throw all image templates
        for temp_img_id, temp_img_data in  self.mde_config_data["images"].items():     
            temp_img_path = os.path.join(self.templates_dir, temp_img_data["path"])   
            #print(f"temp_img_path  .........................{temp_img_path} ")
            #print('current template image size: ', temp_img_data["size"])
            match_values=[]
            features_count=0
            match_count=0
            # loop throw all feature in the current template image
            for merkma_id, feature in temp_img_data["features"].items(): 
                features_count +=1
                position = feature["position"]
                x1, x2, y1, y2=int(position["x1"]), int(position["x2"]),int( position["y1"]),int(position["y2"])
                temp_img= cv2.imread(temp_img_path)

                #resize the image to make sure ...
                img_resized = resize_image_cv2(img,temp_img_data["size"])
                cropped_img = img_resized[y1:y2, x1:x2] 
                cropped_temp = temp_img[y1:y2, x1:x2] 

                cropped_temp_gray =  convert_to_grayscale(cropped_temp)
                imge_cropped_gray =  convert_to_grayscale(cropped_img)
    
                match_val = self.compute_match_value(imge_cropped_gray, cropped_temp_gray )
               
                match_values.append(match_val)
                if match_val>= min_match_val:
                    match_count +=1
            if match_count == features_count:
               # print('*********************************************')
                #print('current match_values')
                #print(f"match_values ={match_values}                temp_img_id = {temp_img_id}     ")
                #print('*********************************************')
                    
                return  match_values,temp_img_id
        return -1,-1
   
    
    def compute_match_value(self, img_gray, template):
        """
        Compute the match value between an input image and a template.

        Parameters:
        - img_gray (ndarray): Grayscale input image.
        - template (ndarray): Template image.

        Returns:
        - match_val (float): Match value.
        """

        try:
            result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            #print(f" result[0][0] = { result[0][0]}")
       
            _, match_val, _, _ = cv2.minMaxLoc(result)
        
            return match_val

        except cv2.error as cv2_error:
            print(f"OpenCV Error: {cv2_error}")
            return None

        except Exception as e:
            print(f"Error: {e}")
            return None



def main():
    # Define paths and parameters
    configFiles_dir = 'ConfigFiles'
    mde_config_file_name = 'mde_config.json'
    templates_dir_name = 'templates'

    # Load the input image for matching
    input_image_path = r"G:\Meine Ablage\test_filter\imge_filter\imge_filter\x1.tiff"
    input_img = cv2.imread(input_image_path)

    # Initialize the ImageMatcher
    matcher = ImageMatcher(configFiles_dir, mde_config_file_name, templates_dir_name)

    # Perform image matching
    match_values, temp_img_id = matcher.match_images(input_img)

    # Process the matching results
    if temp_img_id != -1:
        print(f"Best template match found: {temp_img_id}")
        print(f"Match values: {match_values}")
    else:
        print("No matching template found.")

if __name__ == "__main__":
    main()



