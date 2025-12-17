import PyInstaller.__main__
import os
import shutil
import sys

def build():
    print("Starting build process...")
    
    # Clean previous build
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    # Define build arguments
    args = [
        'app_qt.py',
        '--name=GNN_GeoMod',
        '--windowed',      # No console window
        '--onedir',        # Create a directory
        '--clean',         # Clean cache
        
        # Include source packages
        '--add-data=src;src',
        '--add-data=configs;configs',
        
        # Hidden imports for scientific stack
        '--hidden-import=sklearn.neighbors._typedefs',
        '--hidden-import=sklearn.utils._cython_blas',
        '--hidden-import=sklearn.neighbors._quad_tree',
        '--hidden-import=sklearn.tree._utils',
        '--hidden-import=scipy.special.cython_special',
        '--hidden-import=torch_geometric',
        '--hidden-import=pyvistaqt',
        
        # Exclude unnecessary modules to save space (optional)
        # '--exclude-module=tkinter',
    ]
    
    print(f"Running PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)
    
    # Copy data directory to dist
    print("Copying data directory...")
    src_data = 'data'
    dst_data = os.path.join('dist', 'GNN_GeoMod', 'data')
    if os.path.exists(src_data):
        shutil.copytree(src_data, dst_data)
        print(f"Data copied to {dst_data}")
    
    print("Build complete. Check 'dist/GNN_GeoMod' folder.")

if __name__ == "__main__":
    build()
