# test_pydantic.py
try:
    from pydantic import BaseModel, create_model
    print("✓ Import réussi")
    
    # Test de create_model
    TestModel = create_model('TestModel', name=(str, ...))
    print("✓ create_model fonctionne")
    
    # Test avec FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    print("✓ FastAPI fonctionne")
    
except ImportError as e:
    print(f"✗ Erreur: {e}")
    import sys
    print(f"Python path: {sys.path}")