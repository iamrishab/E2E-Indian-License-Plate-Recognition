import cv2
import pytesseract
from LPRv1.utils import allow_needed_values as anv 


class TesserectRecognizeText(object):
    
    def __init__(self):
        super(TesserectRecognizeText, self).__init__()
        pass

    def recognize_text(self, cropped_image):
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        text = anv.catch_rectify_plate_characters(pytesseract.image_to_string(image_rgb, lang=None))
        return text