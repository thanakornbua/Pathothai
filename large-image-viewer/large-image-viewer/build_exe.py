import os
import sys
import subprocess

def build_executable():
    # Define the command to build the executable using PyInstaller
    command = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        'src/main.py',
        '--distpath',
        'dist',
        '--workpath',
        'build',
        '--specpath',
        'build'
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Executable built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during build: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()