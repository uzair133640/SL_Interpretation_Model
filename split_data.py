import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir='asl_dataset', output_dir='split_data', test_size=0.15, val_size=0.15):
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process each class
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for cls in classes:
        src_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train+val and test
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        # Split train_val into train and val
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        # Copy files to respective directories
        for split, files in [('train', train), ('val', val), ('test', test)]:
            dest_dir = os.path.join(output_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

if __name__ == '__main__':
    split_dataset()