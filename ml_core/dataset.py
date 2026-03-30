import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class xBDDamageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Updated to look for .png instead of .tif
        all_files = os.listdir(images_dir)
        self.base_ids = list(set([
            f.replace('_pre_disaster.png', '').replace('_post_disaster.png', '') 
            for f in all_files if f.endswith('.png')
        ]))

    def __len__(self):
        return len(self.base_ids)

    def _load_image(self, filepath):
        """Loads a PNG using OpenCV and normalizes it."""
        image = cv2.imread(filepath)
        # OpenCV loads in BGR, we must convert to RGB for the AI
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _generate_mask(self, json_path, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not os.path.exists(json_path):
            return mask

        with open(json_path, 'r') as f:
            label_data = json.load(f)
            
        polygons = label_data['features']['xy']
        for poly in polygons:
            damage_type = poly['properties'].get('subtype', 'no-damage')
            if damage_type in ['minor-damage', 'major-damage', 'destroyed']:
                coords = poly['wkt']
                try:
                    raw_coords = coords.replace('POLYGON ((', '').replace('))', '').split(', ')
                    pts = np.array([[float(c) for c in pt.split(' ')] for pt in raw_coords], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                except Exception:
                    pass 
        return mask.astype(np.float32)

    def __getitem__(self, idx):
        base_id = self.base_ids[idx]
        
        # Updated paths to target .png
        pre_img_path = os.path.join(self.images_dir, f"{base_id}_pre_disaster.png")
        post_img_path = os.path.join(self.images_dir, f"{base_id}_post_disaster.png")
        post_label_path = os.path.join(self.labels_dir, f"{base_id}_post_disaster.json")

        pre_img = self._load_image(pre_img_path)
        post_img = self._load_image(post_img_path)

        stacked_img = np.concatenate([pre_img, post_img], axis=-1)

        mask = self._generate_mask(post_label_path, pre_img.shape)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=stacked_img, mask=mask)
            stacked_img = augmented['image']
            mask = augmented['mask']

        tensor_img = torch.from_numpy(stacked_img.transpose(2, 0, 1))
        tensor_mask = torch.from_numpy(mask.transpose(2, 0, 1))

        return tensor_img, tensor_mask