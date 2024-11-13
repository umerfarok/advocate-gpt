import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.append(str(src_dir))

from src.api.server import app

if __name__ == "__main__":
    app.run(debug=True)