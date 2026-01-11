import subprocess
import os
import sys

directory = "/Volumes/2TB_2401BU/Photo_Restore_02/Baby Book"

try:
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.heic', '.jpg', '.jpeg')) and not f.startswith("._")]
except FileNotFoundError:
    print(f"Directory not found: {directory}")
    sys.exit(1)

print(f"Checking {len(files)} files in {directory}...")

for filename in files[:20]: # Check first 20
    filepath = os.path.join(directory, filename)
    try:
        # Use sips to get the orientation property
        result = subprocess.run(['sips', '-g', 'orientation', filepath], capture_output=True, text=True)
        output = result.stdout.strip()
        # Output format is typically:
        # /path/to/file:
        #   orientation: 1 (Normal)
        # or
        #   orientation: 6 (Rotate 90 CW)
        
        lines = output.splitlines()
        orientation_line = next((line for line in lines if "orientation:" in line), None)
        
        if orientation_line:
            orientation_val = orientation_line.split(":")[1].strip()
            print(f"{filename}: Orientation {orientation_val}")
        else:
            print(f"{filename}: Could not find orientation in sips output")

    except Exception as e:
        print(f"{filename}: Error - {e}")
