import sys
from pathlib import Path

# Add the project root to sys.path so that tests can import config, game, level, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))
