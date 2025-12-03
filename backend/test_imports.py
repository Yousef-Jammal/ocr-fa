import sys

print("Testing imports...")
print("=" * 50)

# Test NeMo
try:
    import nemo
    print(f"✓ NeMo version: {nemo.__version__}")
except ImportError as e:
    print(f"✗ NeMo import failed: {e}")

# Test NeMo collections
try:
    import nemo.collections.nlp as nemo_nlp
    print("✓ NeMo NLP collections available")
except ImportError as e:
    print(f"✗ NeMo NLP collections failed: {e}")

# Test Riva client
try:
    import riva.client
    print("✓ NVIDIA Riva client available")
except ImportError as e:
    print(f"✗ Riva client failed: {e}")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch failed: {e}")

# Test transformers
try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers failed: {e}")

# Test audio libraries
try:
    import soundfile
    import librosa
    print("✓ Audio libraries (soundfile, librosa) available")
except ImportError as e:
    print(f"✗ Audio libraries failed: {e}")

# Test custom modules
try:
    from nemo_generator import NeMoTextGenerator
    print("✓ nemo_generator module available")
except ImportError as e:
    print(f"✗ nemo_generator failed: {e}")

# audio_generator removed in v2.0 (text + images only)
print("ℹ audio_generator removed in v2.0")

try:
    from image_generator_v2 import ImageGenerator
    print("✓ image_generator_v2 module available")
except ImportError as e:
    print(f"✗ image_generator failed: {e}")

print("=" * 50)
print("Import test complete!")
