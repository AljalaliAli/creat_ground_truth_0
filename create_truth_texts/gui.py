import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class ImageDisplay:
    def __init__(self):
        """Initialize the ImageDisplay class, set up the root window and layout."""
        # Initialize the Tkinter root window
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Set up window close protocol

        # Set window size to full screen and disable resizing
        self.root.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.resizable(width=False, height=False)
        self.img_container = None  # Container for the image display
        self.entries_container = None  # Container for the table of entries

        # Calculate dimensions and positions for the image and entries containers
        self.entries_container_width = int(self.root.winfo_screenwidth() * 0.1)
        self.entries_container_height = int(self.root.winfo_screenheight() * 0.9)
        self.entries_container_x = int(self.root.winfo_screenwidth() - self.entries_container_width)
        self.entries_container_y = 0
        
        self.img_container_width = int(self.root.winfo_screenwidth() * 0.9)
        self.img_container_height = int(self.root.winfo_screenheight() * 0.9)
        self.img_container_x = 0
        self.img_container_y = 0
        
        self.img_resize_width = int(self.img_container_width)
        self.img_resize_height = int(self.img_container_height)
        
        self.resize_percent_width = None  # Percentage to resize width
        self.resize_percent_height = None  # Percentage to resize height

        # Define color and font variables
        self.img_container_bg_color = '#4E4E6E'
        self.table_frame_bg_color = '#fff'
        self.label_bg_color = '#fff'
        self.label_font = ("Helvetica", 9, "bold")  # Font properties for labels

        # Create the image display, button, and editable table
        self.create_image_label()
        self.next_img_button = None 
        self.config_issue_button = None
        self.create_editable_table()
        self.next_image = False  # Flag to indicate the next image should be displayed
        self.config_issue_button_clicked=False
        
    def create_image_label(self):
        """Create a label to display the image."""
        # Create the image container frame
        self.img_container = tk.Frame(self.root, width=self.img_container_width, height=self.img_container_height, bg=self.img_container_bg_color)
        self.img_container.place(x=self.img_container_x, y=self.img_container_y)

        # Create the canvas to display the image
        self.img_canvas = tk.Canvas(self.img_container, width=self.img_container_width, height=self.img_container_height, cursor="cross")
        self.img_canvas.place(x=0, y=0)

    def create_editable_table(self):
        """Create an initial editable table with 10 rows and 2 columns."""
        # Create the table frame for entries
        self.table_frame = tk.Frame(self.root, width=self.entries_container_width, height=self.entries_container_height, bg=self.table_frame_bg_color)
        self.table_frame.place(x=self.entries_container_x, y=self.entries_container_y)

        self.entries = []  # List to hold entry widgets
        self.labels = []  # List to hold label widgets
        
    def update_table(self, par_names_and_values_dict):
        """Update the table with the contents from the provided dictionary."""
        # Remove all current entries and labels from the grid
        for row_entries in self.entries:
            for entry in row_entries:
                entry.destroy()
                
        for row_labels in self.labels:
            for label in row_labels:
                label.destroy()
        try:  # Destroy the button if it exists
            self.next_img_button.destroy()
            self.config_issue_button.destroy()
        except:
            pass
        self.labels = []
        self.entries = []
        updated_par_names_and_values_dict = {}

        # Create new entries based on the dictionary length
        for i, (key, value) in enumerate(par_names_and_values_dict.items()):
            row_entries = []
            row_labels = []
            for j, text in enumerate([str(key), str(value)]):  # Ensure key and value are strings
                if value is not None:
                    if j == 0:  # Label
                        label = tk.Label(self.table_frame, text=text, bg=self.label_bg_color, font=self.label_font)
                        label.grid(row=i, column=j)
                        row_labels.append(label)
                    else:  # Entry
                        sv = tk.StringVar()
                        sv.trace("w", lambda name, index, mode, sv=sv, key=key: self.on_value_change(sv, key, updated_par_names_and_values_dict))
                        entry = tk.Entry(self.table_frame, textvariable=sv)
                        entry.insert(0, text)
                        entry.grid(row=i, column=j)
                        row_entries.append(entry)
            self.entries.append(row_entries)
            self.labels.append(row_labels)
        
        # Place the button below the last entry
        self.create_next_img_button(len(par_names_and_values_dict))
        self.create_config_issue_button(len(par_names_and_values_dict))
        return updated_par_names_and_values_dict

    def on_value_change(self, sv, key, updated_par_names_and_values_dict):
        """Callback function to be called when an entry's value changes."""
        updated_par_names_and_values_dict[key] = sv.get()
        #print('updated_par_names_and_values_dict ', updated_par_names_and_values_dict)
        #print(f'input : {key}  changed to { updated_par_names_and_values_dict[key]}')

    def display_image(self, img):
        """Display the given image in the image label."""
        # Convert the image from array to PIL image
        original_image = Image.fromarray(np.uint8(img))
        original_width, original_height = original_image.size
        self.original_image_size = {"width": original_width, "height": original_height}

        # Resize the image
        resized_image = original_image.resize((self.img_resize_width, self.img_resize_height))
        resized_width, resized_height = resized_image.size

        # Calculate resize percentage in width and height
        self.resize_percent_width = self.img_resize_width / original_width
        self.resize_percent_height = self.img_resize_height / original_height

        # Convert to Tkinter PhotoImage and display in canvas
        self.img = ImageTk.PhotoImage(resized_image)
        if self.img_canvas:
            self.img_item = self.img_canvas.create_image(0, 0, image=self.img, anchor="nw")

        # Update the canvas and start the Tkinter event loop
        self.img_canvas.config(width=resized_width, height=resized_height)
        self.img_canvas.pack()

    def create_next_img_button(self, row):
        """Create a button that sets the next_image flag to True when clicked."""
        self.next_img_button = tk.Button(self.table_frame, text="Next Image", command=self.do_something)
        self.next_img_button.grid(row=row, column=0, columnspan=2, pady=10)
    
    def create_config_issue_button(self, row):
        """Create a button that moves the current image to the Configuration Issue folder."""
        self.config_issue_button = tk.Button(self.table_frame, text="Configuration Issue",bg="red",  command=self.do_something_2)
        self.config_issue_button.grid(row=row+1, column=0, columnspan=2, pady=500)

    def do_something(self):
        """Set the next_image flag to True when the button is clicked."""
        self.next_image = True
        print('next_image')

    def do_something_2(self):
        """Handle the Configuration Issue button action."""
        self.config_issue_button_clicked=True
        print('config_issue_button_clicked!')

    def on_close(self):
        """Print the table data when the GUI is closed."""
        for row_entries in self.entries:
            row_data = [entry.get() for entry in row_entries]
            print(row_data)
        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

def show(display, img, par_names_and_values_dict, key_to_remove):
    """Update the ImageDisplay with the given image and table data, and wait for the next_image flag to be set."""
    updated_par_names_and_values_dict = display.update_table(par_names_and_values_dict)
    display.display_image(img)
    display.root.update_idletasks()
    display.root.update()
    while not display.next_image and not display.config_issue_button_clicked:
        display.root.update_idletasks()
        display.root.update()
    display.next_image = False
    return updated_par_names_and_values_dict

# Example usage
if __name__ == "__main__":
    display = ImageDisplay()

    # Example data dictionary
    data = {'Row 1': 'Data 1', 'Row 2': 'Data 2', 'Row 3': 'Data 3', 'Row 4': 'Data 4', 'Row 5': 'Data 5'}

    # Example image (replace with actual image data)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Key to remove
    key_to_remove = 'Row 3'

    # Show the display with the image and data, removing the specified key
    show(display, img, data, key_to_remove)
    display.run()
