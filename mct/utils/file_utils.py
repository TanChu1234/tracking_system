"""
File I/O and directory monitoring utility functions for MCT system.
"""
import os
import json
import time


def load_json_file(file_path):
    """Load JSON file, return empty dict if not exists or error."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_file(file_path, data):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def count_directories_and_files(directory):
    """Count number of directories and files in a directory."""
    if not os.path.exists(directory):
        return 0, 0
    
    folder_count = 0
    file_count = 0
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_count += 1
            try:
                file_count += len([f for f in os.listdir(item_path) 
                                 if os.path.isfile(os.path.join(item_path, f))])
            except Exception:
                pass
        elif os.path.isfile(item_path):
            file_count += 1
    
    return folder_count, file_count


def monitor_face_directory(faces_dir, update_queue, check_interval=10):
    """
    Monitor faces directory for changes and trigger reload.
    Runs in separate thread.
    
    Args:
        faces_dir: Path to faces directory
        update_queue: Queue to put update signals
        check_interval: Check every N seconds (default: 10)
    """
    print(f"🔍 Starting face directory monitor (checking every {check_interval}s)...")
    
    initial_folders, initial_files = count_directories_and_files(faces_dir)
    print(f"📊 Initial state - Folders: {initial_folders}, Files: {initial_files}")
    
    while True:
        try:
            time.sleep(check_interval)
            
            current_folders, current_files = count_directories_and_files(faces_dir)
            
            if initial_folders != current_folders or initial_files != current_files:
                print(f"\n{'='*60}")
                print(f"📢 FACE DIRECTORY CHANGE DETECTED!")
                print(f"   Previous: {initial_folders} folders, {initial_files} files")
                print(f"   Current:  {current_folders} folders, {current_files} files")
                print(f"{'='*60}")
                
                update_queue.put('reload_faces')
                
                initial_folders, initial_files = current_folders, current_files
                
        except Exception as e:
            print(f"⚠️ Error in face directory monitor: {e}")
            time.sleep(check_interval)
