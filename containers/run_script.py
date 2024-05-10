import sys
import subprocess

# Get the path to the script from command-line arguments
script_path = sys.argv[1]

# Run the specified script
subprocess.run(sys.argv[1:])