import os

def delete_unmatched_raw(jpg_folder, raw_folder):
    jpg_files = {os.path.splitext(f)[0] for f in os.listdir(jpg_folder) if f.lower().endswith('.jpg')}

    raw_files = [f for f in os.listdir(raw_folder) if f.lower().endswith('.cr2')]

    for raw_file in raw_files:
        raw_name = os.path.splitext(raw_file)[0]

        if raw_name not in jpg_files:
            raw_path = os.path.join(raw_folder, raw_file)
            os.remove(raw_path) 
            print(f"❌ Supprimé : {raw_path}")

    print("✅ Nettoyage terminé !")

jpg_folder = "data/processed"
raw_folder = "data/raw"

delete_unmatched_raw(jpg_folder, raw_folder)
