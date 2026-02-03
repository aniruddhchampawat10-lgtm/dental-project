import os
import shutil

# -------------------------------------------------
# YOUR EXACT PATHS
# -------------------------------------------------
NORMAL_DIR  = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\Dataset\Normal"
CARRIES_DIR = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\Dataset\Carries"

IMAGE_OUT = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\data\images"
MASK_OUT  = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\data\masks"
# -------------------------------------------------

os.makedirs(IMAGE_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

def process_folder(folder_path):
    print("\nProcessing folder:", folder_path)

    if not os.path.exists(folder_path):
        print("Folder NOT found")
        return

    files = os.listdir(folder_path)
    print("Files found:", len(files))

    for file in files:
        src = os.path.join(folder_path, file)

        if not file.lower().endswith(".png"):
            continue

        # MASK FILE
        if file.lower().endswith("-mask.png"):
            new_name = file.replace("-mask", "")
            dst = os.path.join(MASK_OUT, new_name)
            shutil.copy(src, dst)
            print("Mask copied:", new_name)

        # IMAGE FILE
        else:
            dst = os.path.join(IMAGE_OUT, file)
            shutil.copy(src, dst)
            print("Image copied:", file)

process_folder(NORMAL_DIR)
process_folder(CARRIES_DIR)

print("\nDATASET SEGREGATION COMPLETED SUCCESSFULLY")
print("Images saved to:", IMAGE_OUT)
print("Masks saved to :", MASK_OUT)
