import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as transforms

class xBDDataset(Dataset):
    """
    Loads real pre/post disaster imagery from the xBD dataset and dynamically 
    rasterizes the JSON polygon labels into binary training masks.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels') 
        
        # Gather all 'post_disaster' PNG files
        self.post_images = [f for f in os.listdir(self.image_dir) if 'post_disaster.png' in f]
        
        # Standardize images to 256x256 for the Neural Network
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor()
        ])

    def parse_wkt_polygon(self, wkt_string):
        """Converts xView2 WKT string into a list of (x, y) coordinates."""
        coords_str = wkt_string.replace("POLYGON ((", "").replace("POLYGON((", "").replace("))", "")
        points = []
        for pt in coords_str.split(","):
            x, y = pt.strip().split(" ")
            points.append((float(x), float(y)))
        return points

    def __len__(self):
        return len(self.post_images)

    def __getitem__(self, idx):
        # 1. Identify the file names
        post_img_name = self.post_images[idx]
        pre_img_name = post_img_name.replace('post_disaster', 'pre_disaster')
        label_name = post_img_name.replace('.png', '.json') # The corresponding JSON file
        
        # 2. Load the RGB images
        pre_img = Image.open(os.path.join(self.image_dir, pre_img_name)).convert("RGB")
        post_img = Image.open(os.path.join(self.image_dir, post_img_name)).convert("RGB")
        
        # 3. Create a blank black canvas for the ground truth mask
        mask = Image.new('L', pre_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 4. Parse the JSON and draw white polygons over destroyed buildings
        json_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                label_data = json.load(f)
                
            for feature in label_data['features']['xy']:
                damage_type = feature['properties'].get('subtype', 'un-classified')
                
                # We only want the AI to learn what actual damage looks like
                if damage_type in ['destroyed', 'major-damage', 'minor-damage']:
                    wkt = feature['wkt']
                    try:
                        polygon = self.parse_wkt_polygon(wkt)
                        draw.polygon(polygon, outline=1, fill=1) # 1 = Destroyed
                    except:
                        pass # Skip malformed polygons

        # 5. Resize mask to match image patch size and binarize
        mask_resized = mask.resize((256, 256), Image.NEAREST)
        target_np = np.array(mask_resized).astype(np.float32)
        target_tensor = torch.from_numpy(target_np).unsqueeze(0)

        # 6. Apply transforms and stack 6 channels
        pre_tensor = self.transform(pre_img)
        post_tensor = self.transform(post_img)
        stacked_input = torch.cat((pre_tensor, post_tensor), dim=0)

        return stacked_input, target_tensor
