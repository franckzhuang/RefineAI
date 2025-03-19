from pathlib import Path
import rawpy
import imageio
import time
import concurrent.futures

def convert_raw_to_jpeg(raw_folder: str, max_workers=4):
    raw_folder = Path(raw_folder)
    
    jpeg_folder = raw_folder / 'JPEG'
    jpeg_folder.mkdir(exist_ok=True)

    raw_extensions = {'.arw', '.cr2', '.dng', '.nef', '.raw'}
    
    raw_files = [f for f in raw_folder.iterdir() if f.suffix.lower() in raw_extensions]
    
    if not raw_files:
        print("No RAW files found.")
        return

    total_files = len(raw_files)
    print(f"Starting conversion of {total_files} RAW files using {max_workers} workers...\n")

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_raw_to_jpeg_single, raw_file, jpeg_folder / f"{raw_file.stem}.jpg"): raw_file
            for raw_file in raw_files
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            raw_file = futures[future]
            try:
                elapsed_time = future.result()  
                print(f"[{i}/{total_files}] Converted: {raw_file.name} ({elapsed_time:.2f} sec)")
            except Exception as e:
                print(f"Error processing {raw_file}: {e}")

    total_time = time.time() - start_time
    mean_time = total_time / total_files if total_files > 0 else 0

    print(f"\n✅ Total Execution Time: {total_time:.2f} sec")
    print(f"✅ Mean Processing Time per Image: {mean_time:.2f} sec")

def convert_raw_to_jpeg_single(raw_path: str, output_path: str) -> time :
    start_time = time.time()
    try:
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess()
            imageio.imwrite(str(output_path), rgb)  
    except Exception as e:
        print(f"Error converting {raw_path}: {e}")
    return time.time() - start_time  

if __name__ == "__main__":
    raw_folder = "to_convert" 
    max_workers = 8  
    convert_raw_to_jpeg(raw_folder, max_workers)
