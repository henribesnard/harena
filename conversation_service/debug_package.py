"""
🔍 Script de Diagnostic Package intent_rules

Ce script diagnostique pourquoi les fonctions package ne fonctionnent pas
malgré la présence des fichiers JSON.
"""

import json
from pathlib import Path
import sys

# Ajout du path
sys.path.append(str(Path(__file__).parent))

def diagnose_files():
    """Diagnostic des fichiers JSON"""
    print("🔍 DIAGNOSTIC DES FICHIERS JSON")
    print("=" * 50)
    
    intent_rules_dir = Path("intent_rules")
    
    files_to_check = [
        "financial_patterns.json",
        "conversational_patterns.json", 
        "entity_patterns.json"
    ]
    
    for filename in files_to_check:
        file_path = intent_rules_dir / filename
        print(f"\n📁 Checking {filename}:")
        
        if not file_path.exists():
            print(f"  ❌ File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  ✅ File exists and valid JSON")
            print(f"  📊 File size: {file_path.stat().st_size} bytes")
            
            if filename == "financial_patterns.json":
                intents = data.get("intents", {})
                print(f"  💰 Financial intents: {len(intents)}")
                for intent_name in intents.keys():
                    print(f"     - {intent_name}")
            
            elif filename == "conversational_patterns.json":
                intents = data.get("intents", {})
                print(f"  💬 Conversational intents: {len(intents)}")
                for intent_name in intents.keys():
                    print(f"     - {intent_name}")
            
            elif filename == "entity_patterns.json":
                entities = data.get("entity_types", {})
                print(f"  📝 Entity types: {len(entities)}")
                for entity_name in entities.keys():
                    print(f"     - {entity_name}")
                    
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")

def diagnose_package_loading():
    """Diagnostic du chargement du package"""
    print("\n\n🔍 DIAGNOSTIC DU CHARGEMENT PACKAGE")
    print("=" * 50)
    
    try:
        # Import direct du package
        import intent_rules
        print("✅ Package intent_rules imported successfully")
        
        # Test get_default_loader
        print("\n📋 Testing get_default_loader:")
        loader = intent_rules.get_default_loader()
        print(f"  ✅ Loader created")
        print(f"  📊 Financial rules: {len(loader.financial_rules)}")
        print(f"  📊 Conversational rules: {len(loader.conversational_rules)}")
        print(f"  📊 Entity patterns: {len(loader.entity_patterns)}")
        
        # Affichage des règles conversationnelles
        print(f"\n💬 Conversational rules details:")
        for name, rule in loader.conversational_rules.items():
            print(f"     - {name}: category={rule.intent_category}, confidence={rule.confidence}")
        
        # Test get_package_info
        print(f"\n📦 Testing get_package_info:")
        info = intent_rules.get_package_info()
        print(f"  📊 Package info rules count:")
        print(f"     - Financial: {info['rules']['financial_count']}")
        print(f"     - Conversational: {info['rules']['conversational_count']}")
        print(f"     - Entity types: {info['rules']['entity_types_count']}")
        
        # Test quick_detect_intent
        print(f"\n🎯 Testing quick_detect_intent:")
        test_cases = ["bonjour", "salut", "aide", "help", "hello"]
        
        for test_case in test_cases:
            match = intent_rules.quick_detect_intent(test_case)
            if match:
                print(f"  ✅ '{test_case}' -> {match.intent} (conf: {match.confidence:.3f})")
            else:
                print(f"  ❌ '{test_case}' -> NO MATCH")
        
        # Test manual engine
        print(f"\n🔧 Testing manual engine creation:")
        engine = intent_rules.create_rule_engine(loader)
        
        for test_case in test_cases:
            match = engine.match_intent(test_case)
            if match:
                print(f"  ✅ '{test_case}' -> {match.intent} (conf: {match.confidence:.3f})")
            else:
                print(f"  ❌ '{test_case}' -> NO MATCH")
                
    except Exception as e:
        print(f"❌ Error in package loading: {e}")
        import traceback
        traceback.print_exc()

def diagnose_rule_validation():
    """Diagnostic de validation des règles"""
    print("\n\n🔍 DIAGNOSTIC VALIDATION DES RÈGLES")
    print("=" * 50)
    
    try:
        from intent_rules.rule_loader import RuleLoader
        
        # Chargement direct avec path explicite
        rules_dir = Path("intent_rules")
        loader = RuleLoader(rules_dir)
        
        print(f"✅ Direct RuleLoader created")
        print(f"📊 Rules loaded:")
        print(f"   - Financial: {len(loader.financial_rules)}")
        print(f"   - Conversational: {len(loader.conversational_rules)}")
        print(f"   - Entity patterns: {len(loader.entity_patterns)}")
        
        # Validation des règles
        errors = loader.validate_rules()
        print(f"\n🔍 Validation results:")
        
        for category, error_list in errors.items():
            if error_list:
                print(f"   ❌ {category}: {len(error_list)} errors")
                for error in error_list[:3]:  # Afficher les 3 premières erreurs
                    print(f"      - {error}")
            else:
                print(f"   ✅ {category}: no errors")
        
        # Test d'un cas simple
        from intent_rules.rule_engine import create_rule_engine
        engine = create_rule_engine(loader)
        
        print(f"\n🎯 Testing simple cases with direct engine:")
        simple_cases = ["bonjour", "aide"]
        
        for case in simple_cases:
            match = engine.match_intent(case)
            if match:
                print(f"   ✅ '{case}' -> {match.intent}")
            else:
                print(f"   ❌ '{case}' -> NO MATCH")
                
    except Exception as e:
        print(f"❌ Error in rule validation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale de diagnostic"""
    print("🚀 DIAGNOSTIC COMPLET PACKAGE INTENT_RULES")
    print("=" * 70)
    
    # 1. Diagnostic des fichiers
    diagnose_files()
    
    # 2. Diagnostic du chargement package
    diagnose_package_loading()
    
    # 3. Diagnostic validation des règles
    diagnose_rule_validation()
    
    print("\n" + "=" * 70)
    print("🔍 Diagnostic terminé !")

if __name__ == "__main__":
    main()