def check_pytorch_model_compatibility():
    """Check if pytorch_model.pth from HuggingFace is compatible and loadable"""
    from resnet_gru_model import ResNetGRUModel
    import torch
    import os
    
    model_path = 'model/pytorch_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ❌ File not found")
        print(f"   To download: Visit https://huggingface.co/ and download the model")
        return False
    
    try:
        model = ResNetGRUModel(num_classes=1)
        
        # Load checkpoint (HuggingFace format with nested state_dict)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Try strict loading first
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ✅ Compatible & Loadable")
                print(f"   Architecture: ResNet + GRU + Attention")
                print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
                if 'epoch' in checkpoint:
                    print(f"   Training Epoch: {checkpoint['epoch']}")
                if 'val_accuracy' in checkpoint:
                    print(f"   Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
                return True
            except RuntimeError as e:
                # Try non-strict loading
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(missing) == 0:
                    print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ✅ Compatible (non-strict load)")
                    print(f"   Unexpected keys: {len(unexpected)} (can be ignored)")
                    return True
                else:
                    print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ⚠️ Partially Compatible")
                    print(f"   Missing keys: {len(missing)}")
                    print(f"   Unexpected keys: {len(unexpected)}")
                    return False
        else:
            print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ❌ Invalid checkpoint format")
            return False
            
    except Exception as e:
        print(f"\n🤗 HuggingFace Model (pytorch_model.pth): ❌ Error loading: {e}")
        return False

import torch
import os
from src.model import get_model
from src.ensemble import create_ensemble

def check_if_trained(model_type, model_path):
    if not os.path.exists(model_path):
        print(f"{model_type}: ❌ File not found")
        return False

    trained_state = torch.load(model_path, map_location="cpu")

    # Create fresh model
    if "ensemble" in model_type:
        ensemble_type = model_type.replace("ensemble_", "")
        fresh_model = create_ensemble(ensemble_type)
    else:
        fresh_model = get_model(model_type)

    fresh_state = fresh_model.state_dict()

    total_diff = 0.0
    param_count = 0

    for key in fresh_state:
        if key in trained_state:
            diff = torch.sum(torch.abs(fresh_state[key] - trained_state[key])).item()
            total_diff += diff
            param_count += fresh_state[key].numel()

    avg_diff = total_diff / param_count

    if avg_diff < 1e-6:
        print(f"{model_type}: ❌ NOT TRAINED (weights nearly identical)")
        return False
    else:
        print(f"{model_type}: ✅ TRAINED")
        print(f"   Avg Weight Difference: {avg_diff:.6f}")
        return True

def main():
    check_pytorch_model_compatibility()
    print("="*60)
    print("🔍 Model Training Analysis")
    print("="*60)

    models_to_check = [
        ("enhanced", "model/enhanced_quick_trained.pth"),  # Trained model
        ("ensemble_multiscale", "model/ensemble_multiscale_best_trained.pth"),  # Trained model
    ]

    print("\n📊 Checking Model Files:")
    print("-"*60)

    trained_count = 0
    total_count = 0

    # Check pytorch_model first (HuggingFace model)
    print(f"\nChecking: model/pytorch_model.pth (HuggingFace)")
    if check_pytorch_model_compatibility():
        trained_count += 1
    total_count += 1

    # Check other models
    for model_type, model_path in models_to_check:
        print(f"\nChecking: {model_path}")
        if check_if_trained(model_type, model_path):
            trained_count += 1
        total_count += 1

    print("-"*60)
    print(f"\nSummary: {trained_count}/{total_count} models trained")

    # Check dataset
    print("\n📁 Dataset Status:")
    print("-"*60)
    if os.path.exists('elevenlabs_dataset'):
        real_count = 0
        fake_count = 0

        if os.path.exists('elevenlabs_dataset/Original'):
            real_count = len([f for f in os.listdir('elevenlabs_dataset/Original') if f.endswith('.wav')])

        for folder in ['ElevenLabs', 'Tacotron', 'Text To Speech', 'Voice Conversion']:
            if os.path.exists(f'elevenlabs_dataset/{folder}'):
                fake_count += len([f for f in os.listdir(f'elevenlabs_dataset/{folder}') if f.endswith('.wav')])

        print(f"✅ ElevenLabs Dataset")
        print(f"   Real (Original): {real_count} files")
        print(f"   Fake (Others): {fake_count} files")
        print(f"   Total: {real_count + fake_count} files")
    else:
        print("❌ ElevenLabs Dataset not found")

    # Training recommendations
    print("\n💡 Recommendations:")
    
    # Check pytorch_model status
    pytorch_exists = os.path.exists('model/pytorch_model.pth')
    if pytorch_exists:
        print("🤗 HuggingFace PyTorch Model: Available")
        print("   Architecture: ResNet + GRU + Attention")
        print("   Use this for experimental/custom model testing")
    else:
        print("🤗 HuggingFace PyTorch Model: Not downloaded")
        print("   Download from: https://huggingface.co/ (your model URL)")
    
    if trained_count == 0:
        print("❌ No models trained! Run training:")
        print("   python train_ensemble.py")
    elif trained_count == 1:
        print("⚠️ Only 1/2 models trained")
        print("   Train the missing model for complete system")
    else:
        print("✅ All models trained! System ready for production")
        if pytorch_exists:
            print("🤗 HuggingFace model also available for comparison")

    print("\nNext steps:")
    if trained_count < 2:
        print("1. Train missing model")
        print("2. Run app: streamlit run app.py")
    else:
        print("1. 🎉 Run app: streamlit run app.py")
        print("2. 🧪 Test both models with different audio files")
        print("3. 📊 Compare performance between models")
        print("4. 🚀 Deploy with Docker: deploy.bat")

if __name__ == "__main__":
    main()
