import base64
from PIL import Image
import io
import sys

def save_image_from_clipboard():
    try:
        # Try to get image from clipboard
        from PIL import ImageGrab
        im = ImageGrab.grabclipboard()
        if im:
            im.save('map_image.png')
            print("Image saved from clipboard as map_image.png")
            return True
    except Exception as e:
        print(f"Error grabbing from clipboard: {e}")
    return False

# If the image can't be grabbed from clipboard, the user can paste base64 image data
print("This script saves an image to disk for use with map_analyzer.py")
print("Attempting to grab image from clipboard...")

if not save_image_from_clipboard():
    print("Could not grab image from clipboard.")
    print("Please save your image as 'map_image.png' in this directory.") 