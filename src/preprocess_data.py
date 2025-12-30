import os
from PIL import Image
from tqdm import tqdm
import shutil
import glob
from random import Random
from joblib import Parallel, delayed
import pandas as pd
import warnings

# --- CONFIGURATION ---
# Note: ViT models typically expect 224x224 input resolution.
# The original script used 200, but 224 improves compatibility with HuggingFace.
OUTPUT_SIZE  = 224  
CROPSIZE_MIN = 160
CROPSIZE_MAX = 2048
CROPSIZE_RATIO = (5,8)
QF_RANGE = (65, 100)

# Compatibility check for newer Pillow (PIL) versions
if hasattr(Image, 'Resampling'):
    resample_method_high = Image.Resampling.LANCZOS
    resample_method_low = Image.Resampling.BICUBIC
else:
    # Legacy support for older Pillow versions
    resample_method_high = Image.ANTIALIAS
    resample_method_low = Image.CUBIC

def check_img(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp'))
            
def transform(input_dir, output_dir, seed, maximg=None):
    """
    Transforms images in the input directory by performing random operations such as cropping, resizing, and jpeg compression.
    It handles sampling (selecting N images) AND preprocessing in one go.
    """
    print(f'Processing from: {input_dir}')
    print(f' -> Saving to: {output_dir}')
    
    # Cleanup: Remove the destination directory if it exists to ensure a fresh start
    if os.path.isdir(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError:
            # Windows sometimes locks files; ignore error if removal fails
            pass 
    os.makedirs(output_dir, exist_ok=True)
    
    random_gen = Random(seed) # Set random seed 
    
    # Recursively search for images
    # Supports both flat directory structures and subdirectories
    input_search = input_dir if '*' in input_dir else os.path.join(input_dir, '**')
    
    # Search for common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    list_src = []
    for ext in extensions:
        list_src.extend(glob.glob(os.path.join(input_search, ext), recursive=True))
        list_src.extend(glob.glob(os.path.join(input_search, ext.upper()), recursive=True))
    
    list_src = sorted(list(set(list_src))) # Remove duplicates and sort the list
    
    total_found = len(list_src)
    print(f" -> Images found: {total_found}")

    # --- IMAGE SELECTION ---
    if maximg is not None:
        if total_found > maximg:
            print(f" -> Selecting {maximg} randomly...")
            random_gen.shuffle(list_src)
            list_src = list_src[:maximg]
        else:
            print(f" -> WARNING: Only {total_found} images available (requested {maximg}). Using all.")
    
    def process_single_image(index, src):
        # Generate a unique filename to avoid collisions from different subfolders
        base_name = os.path.basename(src)
        filename_dst = f"img_{index:03d}_{base_name}"
        if not filename_dst.lower().endswith(".jpg"):
            filename_dst = filename_dst + ".jpg"

        dst = os.path.join(output_dir, filename_dst)

        try:
            with Image.open(src) as img:
                img = img.convert('RGB')
                width, height = img.size

                # 1. RANDOM CROP
                cropmax = min(min(width, height), CROPSIZE_MAX)
                if cropmax < CROPSIZE_MIN:
                    # If image is too small, resize directly without aggressive cropping
                    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), resample_method_high)
                    img.save(dst, "JPEG", quality=90)
                    return [filename_dst, dst, src, 0, 0, 0, 90]
                
                cropmin = max(cropmax * CROPSIZE_RATIO[0] // CROPSIZE_RATIO[1], CROPSIZE_MIN)
                cropsize = random_gen.randint(cropmin, cropmax)

                # Determine crop position
                x1 = random_gen.randint(0, width - cropsize)
                y1 = random_gen.randint(0, height - cropsize)

                img = img.crop((x1, y1, x1+cropsize, y1+cropsize))

                # 2. RESIZE (To 224x224 for ViT standard)
                interp = resample_method_high if cropsize > OUTPUT_SIZE else resample_method_low
                img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), interp)

                # 3. JPEG COMPRESSION (Data Augmentation)
                qf = random_gen.randint(*QF_RANGE)
                img.save(dst, "JPEG", quality=qf)
            
            return [filename_dst, dst, src, cropsize, x1, y1, qf]
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return None
    
    # Execute processing in parallel
    metainfo = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_single_image)(index, src) for index, src in enumerate(tqdm(list_src))
    )
    
    # Filter out failed processes (None results)
    metainfo = [x for x in metainfo if x is not None]

    # Save metadata to CSV for reference
    metainfo_df = pd.DataFrame(metainfo, columns=['filename','dst','src','cropsize','x1','y1','qf'])
    output_csv = os.path.join(output_dir, "metadata_processing.csv")
    metainfo_df.to_csv(output_csv, index=False)
    print(f" -> Processing complete in: {output_dir}\n")
            
if __name__=='__main__':
    # --- LOCAL PATH CONFIGURATION ---
    # Adjust these paths if your directory structure changes
    base_raw = r"C:\Users\Asus\Desktop\Master\Explicabilidad\final-project-joaquinmaciias\raw_data"
    base_dest = r"C:\Users\Asus\Desktop\Master\Explicabilidad\final-project-joaquinmaciias\dataset_100"

    print("=== STARTING XAI PREPROCESSING ===")
    
    # 1. Process REAL images (Select 50, crop, resize to 224px)
    transform(
        input_dir=os.path.join(base_raw, "artgraph_real"),
        output_dir=os.path.join(base_dest, "real"),
        seed=42,     # Fixed seed for reproducibility
        maximg=50    # Select only 50 images
    )

    # 2. Process FAKE images (Select 50, crop, resize to 224px)
    transform(
        input_dir=os.path.join(base_raw, "artgraph_fake"),
        output_dir=os.path.join(base_dest, "fake"),
        seed=42,
        maximg=50    # Select only 50 images
    )
    
    print("DONE! You can now run the notebook.")