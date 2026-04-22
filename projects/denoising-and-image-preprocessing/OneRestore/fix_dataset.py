import os
import shutil
import random

def fix_and_split():
    # Paths based on your terminal output
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(SCRIPT_DIR, 'image', 'CDD-11_train')
    test_dir = os.path.join(SCRIPT_DIR, 'image', 'CDD-11_test')
    
    classes = [
        'clear', 'low', 'haze', 'blur', 'noise', 
        'low_haze', 'low_blur', 'low_noise', 'haze_blur', 
        'haze_noise', 'low_haze_blur', 'low_haze_noise', 'low_haze_blur_noise'
    ]

    print("1. Re-merging test data back into train data to fix alignment...")
    for c in classes:
        test_c_dir = os.path.join(test_dir, c)
        train_c_dir = os.path.join(train_dir, c)
        if os.path.exists(test_c_dir):
            for f in os.listdir(test_c_dir):
                src = os.path.join(test_c_dir, f)
                dst = os.path.join(train_c_dir, f)
                shutil.move(src, dst)
    
    print("2. Performing Synchronized 80/20 split...")
    # Get master list of files from 'clear'
    master_files = [f for f in os.listdir(os.path.join(train_dir, 'clear')) if f.endswith('.jpg')]
    num_to_move = int(len(master_files) * 0.2)
    
    # Select the EXACT SAME filenames to move for ALL 13 classes
    test_files = random.sample(master_files, num_to_move)
    
    for c in classes:
        train_c_dir = os.path.join(train_dir, c)
        test_c_dir = os.path.join(test_dir, c)
        os.makedirs(test_c_dir, exist_ok=True)
        
        for f in test_files:
            src = os.path.join(train_c_dir, f)
            dst = os.path.join(test_c_dir, f)
            if os.path.exists(src):
                shutil.move(src, dst)
                
    print(f"Done! Perfectly aligned {num_to_move} test images across all 13 classes.")

if __name__ == '__main__':
    fix_and_split()