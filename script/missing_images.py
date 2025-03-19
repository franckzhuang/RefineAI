import os
from glob import glob

def find_missing_images(raw_dir, processed_dir):
    raw_files = glob(os.path.join(raw_dir, "*.CR2")) 
    processed_files = glob(os.path.join(processed_dir, "*.jpg")) 

    raw_names = {os.path.splitext(os.path.basename(f))[0] for f in raw_files}
    processed_names = {os.path.splitext(os.path.basename(f))[0] for f in processed_files}

    missing_names = list(raw_names - processed_names)

    return missing_names


if __name__ == "__main__":
    raw_directory = "data/raw" 
    processed_directory = "data/processed"  

    missing = find_missing_images(raw_directory, processed_directory)

    if missing:
        print("Images manquantes dans le dossier traité:")
        for name in missing:
            print(name)
    else:
        print("Aucune image manquante.")