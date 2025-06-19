"""
Install CLIP for CortexFlow-CLIP-CNN V1 Enhancement
==================================================

Script to install and test CLIP library for pre-trained weights integration.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def test_clip_installation():
    """Test CLIP installation"""
    try:
        import clip
        import torch
        
        print("🔍 Testing CLIP installation...")
        
        # List available models
        available_models = clip.available_models()
        print(f"📋 Available CLIP models: {available_models}")
        
        # Test loading a model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ Successfully loaded ViT-B/32 model")
        
        # Test model properties
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🔢 Visual output dim: {model.visual.output_dim}")
        
        return True
        
    except ImportError as e:
        print(f"❌ CLIP import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🎯 Installing CLIP for CortexFlow-CLIP-CNN V1")
    print("=" * 50)
    
    # Install dependencies
    dependencies = [
        "ftfy",
        "regex", 
        "tqdm",
        "Pillow"
    ]
    
    print("📦 Installing dependencies...")
    for dep in dependencies:
        install_package(dep)
    
    # Install CLIP
    print("\n📦 Installing CLIP...")
    clip_success = install_package("git+https://github.com/openai/CLIP.git")
    
    if not clip_success:
        print("⚠️ Direct CLIP install failed, trying alternative...")
        # Try alternative installation
        try:
            import urllib.request
            import zipfile
            
            print("📥 Downloading CLIP manually...")
            url = "https://github.com/openai/CLIP/archive/main.zip"
            urllib.request.urlretrieve(url, "clip-main.zip")
            
            print("📂 Extracting CLIP...")
            with zipfile.ZipFile("clip-main.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            print("📦 Installing CLIP from source...")
            os.chdir("CLIP-main")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
            os.chdir("..")
            
            print("✅ CLIP installed from source")
            clip_success = True
            
        except Exception as e:
            print(f"❌ Alternative CLIP install failed: {e}")
    
    # Test installation
    if clip_success:
        print("\n🧪 Testing CLIP installation...")
        test_success = test_clip_installation()
        
        if test_success:
            print("\n🎉 CLIP Installation Complete!")
            print("✅ Ready for CortexFlow-CLIP-CNN V1 enhancement")
        else:
            print("\n⚠️ CLIP installed but testing failed")
    else:
        print("\n❌ CLIP installation failed")
        print("💡 You can continue with CLIP-inspired architecture")

if __name__ == "__main__":
    main()
