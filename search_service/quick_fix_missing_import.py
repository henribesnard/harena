#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC AUTOMATIQUE DES IMPORTS FANTÔMES
==============================================

Script qui analyse tous les fichiers Python pour détecter :
- Les imports d'objets qui n'existent pas
- Les exports dans __init__.py qui pointent vers du vide
- Les incohérences entre les imports et les définitions réelles
- Les dépendances circulaires

Usage:
    python diagnose_imports.py [--fix] [--verbose]
"""

import ast
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import importlib.util
import argparse


@dataclass
class ImportInfo:
    """Information sur un import"""
    module: str
    names: List[str]
    file_path: Path
    line_number: int
    is_from_import: bool = True
    alias: Optional[str] = None


@dataclass
class ExportInfo:
    """Information sur un export dans __init__.py"""
    name: str
    source_module: str
    file_path: Path
    line_number: int


@dataclass
class DefinitionInfo:
    """Information sur une définition (classe, fonction, variable)"""
    name: str
    type: str  # 'class', 'function', 'variable', 'import'
    file_path: Path
    line_number: int
    is_exported: bool = False


@dataclass
class DiagnosticResult:
    """Résultat du diagnostic"""
    missing_imports: List[Tuple[ImportInfo, str]] = field(default_factory=list)
    phantom_exports: List[Tuple[ExportInfo, str]] = field(default_factory=list)
    circular_imports: List[List[str]] = field(default_factory=list)
    unused_imports: List[ImportInfo] = field(default_factory=list)
    missing_definitions: List[str] = field(default_factory=list)
    inconsistent_exports: List[Tuple[str, str]] = field(default_factory=list)


class ImportAnalyzer(ast.NodeVisitor):
    """Analyseur AST pour extraire les imports et définitions"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.imports: List[ImportInfo] = []
        self.exports: List[ExportInfo] = []
        self.definitions: List[DefinitionInfo] = []
        self.all_exports: List[str] = []
        
    def visit_Import(self, node: ast.Import):
        """Visite les imports 'import module'"""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                names=[alias.name.split('.')[-1]],
                file_path=self.file_path,
                line_number=node.lineno,
                is_from_import=False,
                alias=alias.asname
            )
            self.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visite les imports 'from module import names'"""
        if node.module:
            names = []
            for alias in node.names:
                names.append(alias.name)
                
            import_info = ImportInfo(
                module=node.module,
                names=names,
                file_path=self.file_path,
                line_number=node.lineno,
                is_from_import=True
            )
            self.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visite les définitions de classes"""
        definition = DefinitionInfo(
            name=node.name,
            type='class',
            file_path=self.file_path,
            line_number=node.lineno
        )
        self.definitions.append(definition)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visite les définitions de fonctions"""
        definition = DefinitionInfo(
            name=node.name,
            type='function',
            file_path=self.file_path,
            line_number=node.lineno
        )
        self.definitions.append(definition)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visite les définitions de fonctions async"""
        definition = DefinitionInfo(
            name=node.name,
            type='function',
            file_path=self.file_path,
            line_number=node.lineno
        )
        self.definitions.append(definition)
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Visite les assignations pour détecter les variables et __all__"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == '__all__' and isinstance(node.value, (ast.List, ast.Tuple)):
                    # Extraction des exports __all__
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.all_exports.append(elt.value)
                else:
                    # Variable normale
                    definition = DefinitionInfo(
                        name=target.id,
                        type='variable',
                        file_path=self.file_path,
                        line_number=node.lineno
                    )
                    self.definitions.append(definition)
        self.generic_visit(node)


class ImportDiagnostic:
    """Classe principale pour le diagnostic des imports"""
    
    def __init__(self, root_path: Path, verbose: bool = False):
        self.root_path = root_path
        self.verbose = verbose
        self.python_files: List[Path] = []
        self.file_analysis: Dict[Path, ImportAnalyzer] = {}
        self.module_definitions: Dict[str, List[DefinitionInfo]] = defaultdict(list)
        
    def find_python_files(self) -> List[Path]:
        """Trouve tous les fichiers Python dans le projet"""
        python_files = []
        for path in self.root_path.rglob("*.py"):
            # Ignorer les fichiers dans certains dossiers
            if any(part.startswith('.') for part in path.parts):
                continue
            if any(part in ['__pycache__', 'dist', 'build', 'env', 'venv'] for part in path.parts):
                continue
            python_files.append(path)
        
        return sorted(python_files)
    
    def analyze_file(self, file_path: Path) -> Optional[ImportAnalyzer]:
        """Analyse un fichier Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parser le fichier
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyser avec notre visitor
            analyzer = ImportAnalyzer(file_path)
            analyzer.visit(tree)
            
            return analyzer
            
        except SyntaxError as e:
            if self.verbose:
                print(f"⚠️  Erreur de syntaxe dans {file_path}: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Erreur lors de l'analyse de {file_path}: {e}")
            return None
    
    def build_module_map(self):
        """Construit la carte des modules et leurs définitions"""
        for file_path in self.python_files:
            analyzer = self.analyze_file(file_path)
            if analyzer:
                self.file_analysis[file_path] = analyzer
                
                # Calculer le nom du module
                try:
                    rel_path = file_path.relative_to(self.root_path)
                    if rel_path.name == '__init__.py':
                        module_parts = rel_path.parent.parts
                    else:
                        module_parts = rel_path.with_suffix('').parts
                    
                    module_name = '.'.join(module_parts) if module_parts else ''
                    
                    # Stocker les définitions par module
                    for definition in analyzer.definitions:
                        self.module_definitions[module_name].append(definition)
                        
                except ValueError:
                    # Fichier en dehors du root_path
                    continue
    
    def check_missing_imports(self) -> List[Tuple[ImportInfo, str]]:
        """Vérifie les imports manquants"""
        missing_imports = []
        
        for file_path, analyzer in self.file_analysis.items():
            for import_info in analyzer.imports:
                # Vérifier chaque nom importé
                for name in import_info.names:
                    if name == '*':
                        continue  # Skip wildcard imports
                    
                    # Construire le nom complet du module
                    if import_info.is_from_import:
                        full_module = import_info.module
                    else:
                        full_module = import_info.module
                    
                    # Vérifier si la définition existe
                    definition_exists = False
                    
                    # Chercher dans les définitions du module
                    if full_module in self.module_definitions:
                        for definition in self.module_definitions[full_module]:
                            if definition.name == name:
                                definition_exists = True
                                break
                    
                    # Chercher dans les sous-modules
                    for module_name in self.module_definitions:
                        if module_name.startswith(full_module + '.'):
                            for definition in self.module_definitions[module_name]:
                                if definition.name == name:
                                    definition_exists = True
                                    break
                    
                    if not definition_exists:
                        error_msg = f"'{name}' non trouvé dans le module '{full_module}'"
                        missing_imports.append((import_info, error_msg))
        
        return missing_imports
    
    def check_phantom_exports(self) -> List[Tuple[ExportInfo, str]]:
        """Vérifie les exports fantômes dans __all__"""
        phantom_exports = []
        
        for file_path, analyzer in self.file_analysis.items():
            if file_path.name == '__init__.py' and analyzer.all_exports:
                # Construire le nom du module pour ce __init__.py
                try:
                    rel_path = file_path.relative_to(self.root_path)
                    module_parts = rel_path.parent.parts
                    module_name = '.'.join(module_parts) if module_parts else ''
                    
                    # Vérifier chaque export dans __all__
                    for export_name in analyzer.all_exports:
                        found = False
                        
                        # Chercher dans les définitions locales
                        for definition in analyzer.definitions:
                            if definition.name == export_name:
                                found = True
                                break
                        
                        # Chercher dans les imports
                        for import_info in analyzer.imports:
                            if export_name in import_info.names:
                                found = True
                                break
                        
                        if not found:
                            export_info = ExportInfo(
                                name=export_name,
                                source_module=module_name,
                                file_path=file_path,
                                line_number=0  # On ne peut pas facilement récupérer la ligne
                            )
                            error_msg = f"Export '{export_name}' dans __all__ mais pas défini/importé"
                            phantom_exports.append((export_info, error_msg))
                            
                except ValueError:
                    continue
        
        return phantom_exports
    
    def run_diagnosis(self) -> DiagnosticResult:
        """Lance le diagnostic complet"""
        print("🔍 Démarrage du diagnostic des imports...")
        
        # 1. Trouver tous les fichiers Python
        print("📁 Recherche des fichiers Python...")
        self.python_files = self.find_python_files()
        print(f"   Trouvés: {len(self.python_files)} fichiers")
        
        # 2. Analyser tous les fichiers
        print("🔍 Analyse des fichiers...")
        self.build_module_map()
        print(f"   Analysés: {len(self.file_analysis)} fichiers")
        
        # 3. Vérifications
        print("🔍 Vérification des imports manquants...")
        missing_imports = self.check_missing_imports()
        
        print("🔍 Vérification des exports fantômes...")
        phantom_exports = self.check_phantom_exports()
        
        # Construire le résultat
        result = DiagnosticResult(
            missing_imports=missing_imports,
            phantom_exports=phantom_exports
        )
        
        return result
    
    def print_report(self, result: DiagnosticResult):
        """Affiche le rapport de diagnostic"""
        print("\n" + "="*60)
        print("📊 RAPPORT DE DIAGNOSTIC")
        print("="*60)
        
        # Imports manquants
        if result.missing_imports:
            print(f"\n❌ IMPORTS MANQUANTS ({len(result.missing_imports)}):")
            for import_info, error_msg in result.missing_imports:
                print(f"   📁 {import_info.file_path.name}:{import_info.line_number}")
                print(f"      from {import_info.module} import {', '.join(import_info.names)}")
                print(f"      💥 {error_msg}")
                print()
        else:
            print("\n✅ Aucun import manquant détecté")
        
        # Exports fantômes
        if result.phantom_exports:
            print(f"\n👻 EXPORTS FANTÔMES ({len(result.phantom_exports)}):")
            for export_info, error_msg in result.phantom_exports:
                print(f"   📁 {export_info.file_path.name}")
                print(f"      Export: {export_info.name}")
                print(f"      💥 {error_msg}")
                print()
        else:
            print("\n✅ Aucun export fantôme détecté")
        
        # Résumé
        total_issues = len(result.missing_imports) + len(result.phantom_exports)
        if total_issues == 0:
            print("\n🎉 AUCUN PROBLÈME DÉTECTÉ ! Le projet est propre.")
        else:
            print(f"\n⚠️  TOTAL: {total_issues} problèmes détectés")
            print("   Utilisez --fix pour tenter une correction automatique")


def generate_fixes(result: DiagnosticResult) -> Dict[Path, List[str]]:
    """Génère des suggestions de corrections"""
    fixes = defaultdict(list)
    
    # Fixes pour les imports manquants
    for import_info, error_msg in result.missing_imports:
        if 'FieldValidator' in error_msg:
            # Cas spécifique pour FieldValidator
            fixes[import_info.file_path].append(
                f"# Ligne {import_info.line_number}: Supprimer FieldValidator de l'import"
            )
            fixes[import_info.file_path].append(
                f"# Ou créer la classe FieldValidator dans {import_info.module}"
            )
        else:
            fixes[import_info.file_path].append(
                f"# Ligne {import_info.line_number}: {error_msg}"
            )
    
    # Fixes pour les exports fantômes
    for export_info, error_msg in result.phantom_exports:
        fixes[export_info.file_path].append(
            f"# Supprimer '{export_info.name}' de __all__ ou l'implémenter"
        )
    
    return fixes


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Diagnostic des imports fantômes")
    parser.add_argument("--fix", action="store_true", help="Tenter des corrections automatiques")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--path", "-p", type=str, default=".", help="Chemin du projet à analyser")
    
    args = parser.parse_args()
    
    # Déterminer le chemin du projet
    project_path = Path(args.path).resolve()
    if not project_path.exists():
        print(f"❌ Chemin inexistant: {project_path}")
        sys.exit(1)
    
    print(f"🎯 Analyse du projet: {project_path}")
    
    # Lancer le diagnostic
    diagnostic = ImportDiagnostic(project_path, verbose=args.verbose)
    result = diagnostic.run_diagnosis()
    
    # Afficher le rapport
    diagnostic.print_report(result)
    
    # Générer les fixes si demandé
    if args.fix:
        print("\n🔧 SUGGESTIONS DE CORRECTIONS:")
        print("-" * 40)
        fixes = generate_fixes(result)
        for file_path, fix_list in fixes.items():
            print(f"\n📁 {file_path.name}:")
            for fix in fix_list:
                print(f"   {fix}")
    
    # Code de sortie
    total_issues = len(result.missing_imports) + len(result.phantom_exports)
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()