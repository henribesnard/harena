# test_simple.py
def test_pydantic_import():
    """Test direct de l'import pydantic"""
    import sys
    print(f"\nPython executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}")
    
    from pydantic import BaseModel, create_model
    assert BaseModel is not None
    assert create_model is not None
    print("✓ Imports OK dans pytest")