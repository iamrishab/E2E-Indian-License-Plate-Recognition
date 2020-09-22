import os
import cv2
import random
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pdb import set_trace

save = 'synth'
num_samples = 100000
os.makedirs(save, exist_ok=True)
# use a truetype font
font1 = ImageFont.truetype("helvetica/Helvetica-Bold-Font.ttf", 100)
font2 = ImageFont.truetype("bebas-neue/BebasNeue Bold.ttf", 100)
font3 = ImageFont.truetype("bebas-neue/BebasNeue Regular.ttf", 100)
font3 = ImageFont.truetype("bebas-neue/BebasNeue Book.ttf", 100)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


for k in range(num_samples):
    lp_part_1 = f"{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
    lp_part_2 = str(random.randint(10, 99))
    lp_part_3 = f"{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
    lp_part_4 = str(random.randint(1000, 9999))
    lp_part_5 = f"{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
                
    part_1, part_2, part_3, part_4 = lp_text = random.sample([lp_part_1, lp_part_2, \
                                                                          lp_part_3, lp_part_4, lp_part_5], 4)
    lp_text = f"{part_1}{random.choice((' ', ''))}{part_2}{random.choice((' ', ''))}{part_3}{random.choice((' ', ''))}{part_4}"
    
    font = random.choice([font1, font2, font3])
    font_width, font_height = font.getsize(lp_text)
    padded_font_width, padded_font_height = int(font_width*1.10), int(font_height*1.35)
    
    bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    txt_color = tuple(abs(255 - x) for x in bg_color)
    img = np.zeros((padded_font_height, padded_font_width, 3), np.uint8) # .fill(bg_color)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((25, 15), lp_text, font=font, fill=txt_color)
    
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # resize image
    scale_percent = random.choice((50, 75, 100)) # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    cv2_img = cv2.resize(cv2_img, dim, interpolation=cv2.INTER_AREA)
    
    # rotate image about the vertical axis
    cv2_img = rotate_image(cv2_img, random.randint(0, 4))
    
    # invert color
    if random.choice((0, 1)):
        cv2_img = cv2.bitwise_not(cv2_img)
        
    cv2.imwrite(os.path.join(save, f"{lp_text}.png"), cv2_img)
