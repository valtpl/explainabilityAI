import sys
import os

# Add parent to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.model_loader import LungCancerModel

def test_load():
    print("Testing LungCancerModel instantiation...")
    model_wrapper = LungCancerModel("LungCancer_Best")
    
    if model_wrapper.model is not None:
        print("SUCCESS: Model loaded.")
        print(f"Model object type: {type(model_wrapper.model)}")
    else:
        print("FAILURE: Model failed to load.")

if __name__ == "__main__":
    test_load()
