import sys
from typing import Dict, TypedDict
from pathlib import Path

# Project root directory
BASE = Path(__file__).parent.parent.parent  # ./

# Add project root to path for both direct execution and module imports
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

class FileData(TypedDict):
    path: Path
    file_size: int
    file_content: str

def fetch_data_files(extension: Path | str | None = 'training_data/') -> Dict[str, 'FileData']:
    # Return the files stored under base/extension
    if extension is None:
        extension = 'training_data/'
    p = BASE / extension

    print(f"Fetching data from {p.absolute}")
    return_data = {}
    
    for file_path in p.iterdir():
        if not file_path.is_file():
            continue
        file_size = file_path.stat().st_size
        file_content = file_path.read_text()
        file_name = file_path.stem

        file_data = FileData(path=file_path, file_size=file_size, file_content=file_content)
        return_data[file_name] = file_data

    return return_data

if __name__ == "__main__":
    data = fetch_data_files()
    for k, v in data.items():
        print(f"{k}: ({v['path']}, {v['file_size']})")