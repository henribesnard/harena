"""
Template Engine pour Conversation Service v2.0
Compilation et parameterization des templates de requêtes JSON
"""

import json
import re
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from jinja2 import Template, Environment, FileSystemLoader, meta
from jinja2.exceptions import TemplateError

logger = logging.getLogger(__name__)

@dataclass
class CompiledTemplate:
    """Template compilé avec métadonnées"""
    name: str
    template_data: Dict[str, Any]
    jinja_template: Template
    parameter_mappings: Dict[str, Any]
    compiled_at: datetime
    cache_duration: timedelta
    file_path: Path

@dataclass
class TemplateCompilationResult:
    """Résultat de compilation d'un template"""
    success: bool
    template: Optional[CompiledTemplate] = None
    compilation_time_ms: float = 0.0
    error_message: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

class TemplateEngine:
    """Moteur de templates pour génération de requêtes search_service"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "templates" / "query"
        self.compiled_templates: Dict[str, CompiledTemplate] = {}
        self.jinja_env = Environment()
        
        # Ajout de fonctions helpers pour les templates
        self.jinja_env.globals['to_es_amount_filter'] = self._to_elasticsearch_amount_filter
        self.jinja_env.globals['is_defined_and_not_none'] = self._is_defined_and_not_none

        # Variables globales pour les templates
        self.jinja_env.globals['all_months'] = self._get_all_months()
        
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "compilations": 0
        }
        
        logger.info(f"TemplateEngine initialized with directory: {self.templates_dir}")

    def _get_all_months(self) -> Dict[str, int]:
        """Retourne le mapping combiné des mois français + anglais"""
        french_months = {
            "janvier": 1, "février": 2, "mars": 3, "avril": 4,
            "mai": 5, "juin": 6, "juillet": 7, "août": 8,
            "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
        }

        english_months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        return {**french_months, **english_months}

    async def initialize(self) -> bool:
        """Initialise le moteur de templates"""
        try:
            if not self.templates_dir.exists():
                logger.error(f"Templates directory not found: {self.templates_dir}")
                return False
                
            # Précompiler tous les templates
            await self._precompile_all_templates()
            logger.info(f"Template engine initialized with {len(self.compiled_templates)} templates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize template engine: {e}")
            return False

    async def load_template(self, intent_group: str, intent_subtype: str) -> Optional[CompiledTemplate]:
        """Charge un template par intention"""
        template_key = f"{intent_group}.{intent_subtype}"
        
        # Vérifier le cache d'abord
        if template_key in self.compiled_templates:
            template = self.compiled_templates[template_key]
            if self._is_template_cache_valid(template):
                self.cache_stats["hits"] += 1
                return template
            else:
                # Template expiré, le supprimer du cache
                del self.compiled_templates[template_key]
        
        # Cache miss - compiler le template
        self.cache_stats["misses"] += 1
        return await self._compile_template_by_intent(intent_group, intent_subtype)

    async def compile_template(self, template_path: Path) -> TemplateCompilationResult:
        """Compile un template depuis un fichier"""
        start_time = time.perf_counter()
        
        try:
            if not template_path.exists():
                return TemplateCompilationResult(
                    success=False,
                    error_message=f"Template file not found: {template_path}"
                )
            
            # Charger le fichier JSON
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Valider la structure du template
            validation_errors = self._validate_template_structure(template_data)
            if validation_errors:
                return TemplateCompilationResult(
                    success=False,
                    validation_errors=validation_errors,
                    error_message=f"Template validation failed: {'; '.join(validation_errors)}"
                )
            
            # Compiler le template Jinja2
            parameters_json = json.dumps(template_data["parameters"])
            jinja_template = Template(parameters_json)
            
            # Calculer la durée de cache
            optimizations = template_data.get("optimizations", {})
            cache_duration = self._parse_cache_duration(optimizations.get("cache_duration", "1h"))
            
            # Créer le template compilé
            compiled_template = CompiledTemplate(
                name=template_data["template_name"],
                template_data=template_data,
                jinja_template=jinja_template,
                parameter_mappings=template_data.get("parameter_mappings", {}),
                compiled_at=datetime.now(),
                cache_duration=cache_duration,
                file_path=template_path
            )
            
            compilation_time = (time.perf_counter() - start_time) * 1000
            self.cache_stats["compilations"] += 1
            
            logger.debug(f"Template compiled successfully: {template_data['template_name']} ({compilation_time:.2f}ms)")
            
            return TemplateCompilationResult(
                success=True,
                template=compiled_template,
                compilation_time_ms=compilation_time
            )
            
        except json.JSONDecodeError as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Invalid JSON in template: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except TemplateError as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Jinja2 template error: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        except Exception as e:
            return TemplateCompilationResult(
                success=False,
                error_message=f"Unexpected error compiling template: {e}",
                compilation_time_ms=(time.perf_counter() - start_time) * 1000
            )

    async def render_template(self, template: CompiledTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rend un template avec les paramètres fournis"""
        try:
            # Préparer les données pour le template
            template_context = self._prepare_template_context(template, parameters)
            
            # Rendre le template
            rendered_json = template.jinja_template.render(**template_context)
            
            # Parser le JSON résultant
            rendered_query = json.loads(rendered_json)
            
            # Post-traitement
            cleaned_query = self._post_process_query(template, rendered_query)
            
            # Restaurer les types corrects des paramètres
            typed_query = self._restore_parameter_types(template, cleaned_query)
            
            return typed_query
            
        except Exception as e:
            logger.error(f"Error rendering template {template.name}: {e}")
            raise

    def _prepare_template_context(self, template: CompiledTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare le contexte pour le template Jinja2"""
        context = {}
        
        for param_name, param_config in template.parameter_mappings.items():
            value = self._extract_parameter_value(param_config, parameters)
            context[param_name] = value
            
        return context

    def _extract_parameter_value(self, param_config: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
        """Extrait une valeur de paramètre selon sa configuration"""
        source = param_config.get("source", "")
        required = param_config.get("required", False)
        default_value = param_config.get("default")
        param_type = param_config.get("type", "string")
        
        # Parser le chemin source (ex: "entities.periode_temporelle.date")
        value = self._get_nested_value(parameters, source)
        
        # Transformer les entités de période relative en objets de date
        if param_config.get("type") == "object" and isinstance(value, str):
            value = self._transform_date_range(value)

        # Transformer les objets montant en filtres Elasticsearch
        if param_config.get("type") == "object" and isinstance(value, dict) and "operator" in value:
            value = self._to_elasticsearch_amount_filter(value)
        
        # Appliquer la valeur par défaut si nécessaire
        if value is None and default_value is not None:
            value = default_value
        
        # Validation requis
        if required and value is None:
            raise ValueError(f"Required parameter missing: {source}")
        
        # Validation du type
        if value is not None:
            value = self._cast_parameter_type(value, param_type)
            
        # Validation des valeurs autorisées
        allowed_values = param_config.get("allowed_values")
        if allowed_values and value not in allowed_values:
            if required:
                raise ValueError(f"Parameter {source} value '{value}' not in allowed values: {allowed_values}")
            else:
                value = default_value
                
        return value

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Récupère une valeur nested par chemin (ex: 'entities.date.gte')"""
        if not path:
            return None
            
        parts = path.split('.')
        current = data
        
        try:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None

    def _cast_parameter_type(self, value: Any, param_type: str) -> Any:
        """Cast une valeur vers le type spécifié"""
        if value is None or value == "None":
            return None

        try:
            if param_type == "integer":
                return int(value)
            elif param_type == "float":
                return float(value)
            elif param_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif param_type == "string":
                # Si la valeur est déjà une liste ou un dict, ne pas la convertir en string
                if isinstance(value, (list, dict)):
                    logger.info(f"⚠️ Type mismatch: param_type is 'string' but value is {type(value).__name__}, keeping original type")
                    return value
                return str(value)
            elif param_type == "merchant_name":
                # Normalisation pour les noms de marchands (première lettre majuscule)
                # Mais ne pas casser les listes !
                if isinstance(value, list):
                    return value
                return str(value).strip().title() if value else value
            elif param_type == "object":
                # Pour les objets, on garde tel quel - la transformation se fera dans _remove_null_values
                return value
            else:
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast value {value} to type {param_type}: {e}")
            return value

    def _post_process_query(self, template: CompiledTemplate, query: Dict[str, Any]) -> Dict[str, Any]:
        """Post-traite la requête générée"""
        # Nettoyage selon les règles du template
        filter_cleanup = template.template_data.get("filter_cleanup", {})

        if filter_cleanup.get("remove_null_values", False):
            query = self._remove_null_values(query)

        if filter_cleanup.get("remove_empty_objects", False):
            query = self._remove_empty_objects(query)

        # Parsing forcé des strings Python en objets natifs (listes, dicts)
        query = self._parse_python_strings(query)

        # Log pour debugging
        if "filters" in query and "merchant_name" in query["filters"]:
            merchant_value = query["filters"]["merchant_name"]
            logger.info(f"🔍 After parsing - merchant_name type: {type(merchant_value).__name__}, value: {merchant_value}")

        # Ajouter automatiquement les agrégations dynamiques si pas déjà présentes
        query = self._build_dynamic_aggregations(query)

        return query

    def _transform_date_range(self, date_range_string: str) -> Dict[str, str]:
        """Transforme une période relative ou absolue en objet de date avec gte/lte"""
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta
        import calendar
        import re

        today = date.today()
        current_year = today.year

        # Normaliser le texte (français et anglais)
        date_range_lower = date_range_string.lower().strip()

        # === VALIDATION DES DATES INVALIDES ===
        # Détecter et corriger les dates ISO invalides comme "2025-02-32"
        iso_invalid_match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_range_string.strip())
        if iso_invalid_match:
            year, month, day = map(int, iso_invalid_match.groups())
            try:
                # Tester si la date est valide
                test_date = date(year, month, day)
                # Si valide, continuer normalement
            except ValueError:
                # Date invalide, corriger automatiquement
                import calendar
                _, max_day = calendar.monthrange(year, month)
                corrected_day = min(day, max_day)
                corrected_date = date(year, month, corrected_day)
                logger.warning(f"Date invalide corrigée: {date_range_string} → {corrected_date.isoformat()}")
                return {
                    "gte": corrected_date.isoformat(),
                    "lte": corrected_date.isoformat()
                }

        # === PLAGES RELATIVES PRÉDÉFINIES ===
        if date_range_lower in ["this_month", "ce mois", "this month"]:
            start_of_month = today.replace(day=1)
            _, last_day = calendar.monthrange(today.year, today.month)
            end_of_month = today.replace(day=last_day)
            return {
                "gte": start_of_month.isoformat(),
                "lte": end_of_month.isoformat()
            }

        elif date_range_lower in ["last_month", "mois dernier", "le mois dernier", "last month"]:
            first_last_month = (today.replace(day=1) - relativedelta(months=1))
            _, last_day = calendar.monthrange(first_last_month.year, first_last_month.month)
            end_last_month = first_last_month.replace(day=last_day)
            return {
                "gte": first_last_month.isoformat(),
                "lte": end_last_month.isoformat()
            }

        elif date_range_lower in ["this_week", "cette semaine", "this week"]:
            days_since_monday = today.weekday()
            start_of_week = today - relativedelta(days=days_since_monday)
            end_of_week = start_of_week + relativedelta(days=6)
            return {
                "gte": start_of_week.isoformat(),
                "lte": end_of_week.isoformat()
            }

        elif date_range_lower in ["last_week", "la semaine dernière", "semaine dernière", "last week"]:
            days_since_monday = today.weekday()
            start_of_last_week = today - relativedelta(days=days_since_monday + 7)
            end_of_last_week = start_of_last_week + relativedelta(days=6)
            return {
                "gte": start_of_last_week.isoformat(),
                "lte": end_of_last_week.isoformat()
            }

        elif date_range_lower in ["today", "aujourd'hui", "aujourd hui", "ce jour"]:
            return {
                "gte": today.isoformat(),
                "lte": today.isoformat()
            }

        elif date_range_lower in ["yesterday", "hier"]:
            yesterday = today - relativedelta(days=1)
            return {
                "gte": yesterday.isoformat(),
                "lte": yesterday.isoformat()
            }

        # === MAPPING DES MOIS ===
        french_months = {
            "janvier": 1, "février": 2, "mars": 3, "avril": 4,
            "mai": 5, "juin": 6, "juillet": 7, "août": 8,
            "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
        }

        english_months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        # Mapping combiné français + anglais
        all_months = {**french_months, **english_months}

        # === DÉTECTION DE PLAGES : "X au Y" ou "X - Y" ===
        range_patterns = [
            r'(\d{1,2}|premier|1er|première|1ère)\s+(au|à|-)\s+(\d{1,2})\s+(\w+)(?:\s+(\d{4}))?',  # "14 au 15 mai", "14-15 mai"
            r'(\d{1,2}|premier|1er|première|1ère)\s*-\s*(\d{1,2})\s+(\w+)(?:\s+(\d{4}))?',  # "02-15 décembre"
            r'(\d{1,2}|premier|1er|première|1ère)\s+(\w+)\s+(au|à|-)\s+(\d{1,2})\s+(\w+)(?:\s+(\d{4}))?',  # "14 mai au 15 juin"
            r'(\d{1,2})/(\d{1,2})\s+(au|à|-)\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?',  # "10/12 au 15/12"
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})\s+(au|à|-)\s+(\d{1,2})-(\d{1,2})-(\d{2,4})',  # "10-12-2024 au 15-12-2024"
        ]

        for pattern in range_patterns:
            match = re.match(pattern, date_range_lower)
            if match:
                return self._parse_date_range_match(match, pattern, french_months, current_year, all_months)

        # === DÉTECTION DE DATES SIMPLES ===
        simple_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # "2024-12-10" (ISO) - en premier!
            r'(\d{1,2}|premier|1er|première|1ère)\s+(\w+)(?:\s+(\d{4}))?',  # "premier mai", "15 avril"
            r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?',  # "10/12", "10/12/2024"
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',  # "10-12-2024"
        ]

        for pattern in simple_patterns:
            match = re.match(pattern, date_range_lower)
            if match:
                parsed_date = self._parse_single_date_match(match, pattern, french_months, current_year, all_months)
                if parsed_date:
                    return {
                        "gte": parsed_date.isoformat(),
                        "lte": parsed_date.isoformat()
                    }

        # === FORMAT YYYY-MM ===
        if re.match(r'^\d{4}-\d{2}$', date_range_string.strip()):
            try:
                year_str, month_str = date_range_string.strip().split('-')
                target_year = int(year_str)
                target_month = int(month_str)
                start_of_target_month = date(target_year, target_month, 1)
                _, last_day = calendar.monthrange(target_year, target_month)
                end_of_target_month = date(target_year, target_month, last_day)
                return {
                    "gte": start_of_target_month.isoformat(),
                    "lte": end_of_target_month.isoformat()
                }
            except (ValueError, calendar.IllegalMonthError):
                pass  # Continue vers le fallback

        # === MOIS SEUL (fallback) ===
        if date_range_lower in all_months:
            target_month = all_months[date_range_lower]
            # Logique contextuelle : si le mois demandé est dans le futur (après le mois actuel), c'est l'année précédente
            # Exemple: septembre 2025, "mai" = 2025 (passé), "octobre" = 2024 (éviter futur)
            target_year = current_year - 1 if target_month > today.month else current_year
            start_of_target_month = date(target_year, target_month, 1)
            _, last_day = calendar.monthrange(target_year, target_month)
            end_of_target_month = date(target_year, target_month, last_day)
            return {
                "gte": start_of_target_month.isoformat(),
                "lte": end_of_target_month.isoformat()
            }

        # Si aucun pattern ne correspond, retourner tel quel (will fail in ES, but preserved for debug)
        return {"gte": date_range_string, "lte": date_range_string}

    def _parse_date_range_match(self, match, pattern, french_months, current_year, all_months):
        """Parse un match de plage de dates et retourne gte/lte"""
        from datetime import date
        import calendar

        groups = match.groups()

        # Pattern 1: "14 au 15 mai" -> groups = ('14', 'au', '15', 'mai', None)
        # Pattern 2: "02-15 décembre" -> groups = ('02', '15', 'décembre', None)
        if len(groups) >= 3:
            # Pour "02-15 décembre", pas de connector explicite
            if len(groups) == 4 and groups[2] in all_months:  # "02-15 décembre"
                day1_str, day2_str, month_str, year_str = groups
                connector = "-"
            elif len(groups) >= 4:  # "14 au 15 mai"
                day1_str, connector, day2_str, month_str = groups[:4]
                year_str = groups[4] if len(groups) > 4 and groups[4] else None
            else:
                return None

            if month_str in all_months:
                target_month = all_months[month_str]
                target_year = int(year_str) if year_str else current_year

                # Ajuster l'année avec logique contextuelle si pas explicite
                if not year_str and target_month > date.today().month:
                    target_year = current_year - 1

                day1 = 1 if day1_str in ["premier", "1er", "première", "1ère"] else int(day1_str)
                day2 = int(day2_str)

                # Validation des jours
                _, last_day = calendar.monthrange(target_year, target_month)
                day1 = min(day1, last_day)
                day2 = min(day2, last_day)

                start_date = date(target_year, target_month, day1)
                end_date = date(target_year, target_month, day2)

                return {
                    "gte": start_date.isoformat(),
                    "lte": end_date.isoformat()
                }

        # Pattern 2: "14 mai au 15 juin" -> groups = ('14', 'mai', 'au', '15', 'juin', None)
        elif len(groups) >= 5 and groups[1] in all_months and groups[4] in all_months:
            day1_str, month1_str, connector, day2_str, month2_str = groups[:5]
            year_str = groups[5] if len(groups) > 5 and groups[5] else None

            target_month1 = all_months[month1_str]
            target_month2 = all_months[month2_str]
            target_year = int(year_str) if year_str else current_year

            # Ajuster l'année avec logique contextuelle si pas explicite
            if not year_str and target_month1 > date.today().month:
                target_year = current_year - 1

            day1 = 1 if day1_str in ["premier", "1er", "première", "1ère"] else int(day1_str)
            day2 = int(day2_str)

            # Validation des jours
            _, last_day1 = calendar.monthrange(target_year, target_month1)
            _, last_day2 = calendar.monthrange(target_year, target_month2)
            day1 = min(day1, last_day1)
            day2 = min(day2, last_day2)

            start_date = date(target_year, target_month1, day1)
            end_date = date(target_year, target_month2, day2)

            return {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }

        # Pattern 3: "10/12 au 15/12" -> groups = ('10', '12', 'au', '15', '12', None)
        elif '/' in pattern and len(groups) >= 5:
            day1, month1, connector, day2, month2 = groups[:5]
            year_str = groups[5] if len(groups) > 5 and groups[5] else None
            target_year = int(year_str) if year_str else current_year

            # Si année sur 2 chiffres, l'ajuster
            if year_str and len(year_str) == 2:
                target_year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)

            start_date = date(target_year, int(month1), int(day1))
            end_date = date(target_year, int(month2), int(day2))

            return {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }

        # Pattern 4: "10-12-2024 au 15-12-2024" -> groups = ('10', '12', '2024', 'au', '15', '12', '2024')
        elif '-' in pattern and len(groups) >= 7:
            day1, month1, year1, connector, day2, month2, year2 = groups[:7]

            # Si année sur 2 chiffres, l'ajuster
            target_year1 = int(year1)
            target_year2 = int(year2)
            if len(year1) == 2:
                target_year1 = 2000 + int(year1) if int(year1) < 50 else 1900 + int(year1)
            if len(year2) == 2:
                target_year2 = 2000 + int(year2) if int(year2) < 50 else 1900 + int(year2)

            start_date = date(target_year1, int(month1), int(day1))
            end_date = date(target_year2, int(month2), int(day2))

            return {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }

        return None

    def _parse_single_date_match(self, match, pattern, french_months, current_year, all_months):
        """Parse un match de date simple et retourne un objet date"""
        from datetime import date
        import calendar

        groups = match.groups()

        # Pattern ISO: "2024-12-10" (doit être testé en premier)
        if pattern == r'(\d{4})-(\d{1,2})-(\d{1,2})':
            year, month, day = groups[:3]
            return date(int(year), int(month), int(day))

        # Pattern 1: "premier mai", "15 avril"
        elif len(groups) >= 2 and groups[1] in all_months:
            day_str, month_str, year_str = groups[:3]
            target_month = all_months[month_str]
            target_year = int(year_str) if year_str else current_year

            # Ajuster l'année avec logique contextuelle si pas explicite
            if not year_str and target_month > date.today().month:
                target_year = current_year - 1

            day = 1 if day_str in ["premier", "1er", "première", "1ère"] else int(day_str)

            # Validation du jour
            _, last_day = calendar.monthrange(target_year, target_month)
            day = min(day, last_day)

            return date(target_year, target_month, day)

        # Pattern 2: "10/12", "10/12/2024"
        elif '/' in pattern:
            if len(groups) == 3:  # jour/mois/année
                day, month, year = groups
                target_year = int(year) if year else current_year

                # Si année sur 2 chiffres, l'ajuster
                if year and len(year) == 2:
                    target_year = 2000 + int(year) if int(year) < 50 else 1900 + int(year)

                return date(target_year, int(month), int(day))

        # Pattern 3: "10-12-2024"
        elif pattern.count('-') == 2:
            day, month, year = groups[:3]
            target_year = int(year)

            # Si année sur 2 chiffres, l'ajuster
            if len(year) == 2:
                target_year = 2000 + int(year) if int(year) < 50 else 1900 + int(year)

            return date(target_year, int(month), int(day))


        return None

    def _build_dynamic_aggregations(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construit des agrégations dynamiques intelligentes selon les spécifications Harena.

        Logique de priorisation :
        1. Analyse du contenu des résultats → Détermine les types de transactions présents
        2. Filtre transaction_type → Limite les types d'agrégation (si appliqué)
        3. Filtre temporel → Détermine la segmentation temporelle
        4. Autres filtres → Définissent les dimensions d'agrégation
        """

        # Ne pas ajouter d'agrégations si déjà présentes ou si user_id absent
        if "aggregations" in query or "user_id" not in query:
            return query

        filters = query.get("filters", {})

        # === ÉTAPE 1: ANALYSE DES FILTRES ===
        aggregation_context = self._analyze_aggregation_context(filters)

        # === ÉTAPE 2: CONSTRUCTION DES AGRÉGATIONS ===
        aggregations = {}

        # Toujours ajouter le compteur de transactions
        aggregations["transaction_count"] = {
            "value_count": {"field": "transaction_id"}
        }

        # === ÉTAPE 3: AGRÉGATIONS PAR TYPE DE TRANSACTION ===
        self._add_transaction_type_aggregations(aggregations, aggregation_context)

        # === ÉTAPE 4: AGRÉGATIONS PAR CATÉGORIE (si applicable) ===
        if aggregation_context["categories"]:
            self._add_category_aggregations(aggregations, aggregation_context)

        # === ÉTAPE 4bis: AGRÉGATIONS PAR MARCHAND (si applicable) ===
        if aggregation_context["merchants"]:
            self._add_merchant_aggregations(aggregations, aggregation_context)

        # === ÉTAPE 5: SEGMENTATION TEMPORELLE (si applicable) ===
        if aggregation_context["temporal_segmentation"]:
            self._add_temporal_aggregations(aggregations, aggregation_context)

        # === ÉTAPE 6: OPTIMISATIONS ET LIMITATIONS ===
        aggregations = self._optimize_aggregations(aggregations)

        query["aggregations"] = aggregations

        # Log pour debugging
        agg_summary = list(aggregations.keys())
        logger.info(f"🎯 Agrégation dynamique: {', '.join(agg_summary)} ({len(agg_summary)} total)")

        return query

    def _analyze_aggregation_context(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les filtres pour déterminer le contexte d'agrégation"""
        from datetime import datetime, date
        from dateutil.relativedelta import relativedelta

        context = {
            "transaction_type_filter": filters.get("transaction_type"),
            "categories": [],
            "merchants": [],
            "temporal_segmentation": False,
            "temporal_period_months": 0,
            "date_range": None,
            "requires_both_types": False  # Sera déterminé plus tard par l'analyse des résultats
        }

        # Analyse des catégories
        category_filter = filters.get("category_name")
        if category_filter:
            if isinstance(category_filter, list):
                context["categories"] = category_filter
            elif isinstance(category_filter, str):
                context["categories"] = [category_filter]

        # Analyse des marchands
        merchant_filter = filters.get("merchant_name")
        if merchant_filter:
            logger.info(f"🔎 _analyze_aggregation_context - merchant_filter type: {type(merchant_filter).__name__}, value: {merchant_filter}")
            if isinstance(merchant_filter, list):
                context["merchants"] = merchant_filter
            elif isinstance(merchant_filter, str):
                context["merchants"] = [merchant_filter]
                logger.warning(f"⚠️ merchant_filter is a string, not a list! Converting to list: {[merchant_filter]}")

        # Analyse temporelle
        date_filter = filters.get("date")
        if date_filter and isinstance(date_filter, dict):
            gte_str = date_filter.get("gte")
            lte_str = date_filter.get("lte")

            if gte_str and lte_str:
                try:
                    # Parser les dates ISO
                    start_date = datetime.fromisoformat(gte_str).date()
                    end_date = datetime.fromisoformat(lte_str).date()

                    # Calculer la différence en mois
                    months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

                    context["date_range"] = {
                        "start": start_date,
                        "end": end_date,
                        "months_diff": months_diff
                    }
                    context["temporal_period_months"] = months_diff

                    # Segmentation temporelle si >= 2 mois
                    context["temporal_segmentation"] = months_diff >= 2

                except (ValueError, TypeError):
                    # Si parsing échoue, pas de segmentation temporelle
                    pass

        return context

    def _add_transaction_type_aggregations(self, aggregations: Dict[str, Any], context: Dict[str, Any]):
        """Ajoute les agrégations par type de transaction selon le contexte"""

        transaction_type_filter = context["transaction_type_filter"]

        if transaction_type_filter == "debit":
            # Seulement les débits
            aggregations["total_debit"] = {
                "filter": {"term": {"transaction_type": "debit"}},
                "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
            }

        elif transaction_type_filter == "credit":
            # Seulement les crédits
            aggregations["total_credit"] = {
                "filter": {"term": {"transaction_type": "credit"}},
                "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
            }

        else:
            # Pas de filtre spécifique → les deux types (sera optimisé plus tard selon les résultats)
            aggregations.update({
                "total_debit": {
                    "filter": {"term": {"transaction_type": "debit"}},
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                },
                "total_credit": {
                    "filter": {"term": {"transaction_type": "credit"}},
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }
            })

    def _add_category_aggregations(self, aggregations: Dict[str, Any], context: Dict[str, Any]):
        """Ajoute les agrégations par catégorie"""

        categories = context["categories"]
        transaction_type_filter = context["transaction_type_filter"]

        for category in categories:
            category_key = category.lower().replace(" ", "_").replace("-", "_")

            if transaction_type_filter == "debit":
                aggregations[f"{category_key}_debit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                {"term": {"category_name.keyword": category}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            elif transaction_type_filter == "credit":
                aggregations[f"{category_key}_credit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                {"term": {"category_name.keyword": category}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            else:
                # Les deux types pour cette catégorie
                aggregations[f"{category_key}_debit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                {"term": {"category_name.keyword": category}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }
                aggregations[f"{category_key}_credit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                {"term": {"category_name.keyword": category}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

    def _add_merchant_aggregations(self, aggregations: Dict[str, Any], context: Dict[str, Any]):
        """Ajoute les agrégations par marchand"""

        merchants = context["merchants"]
        transaction_type_filter = context["transaction_type_filter"]

        for merchant in merchants:
            merchant_key = merchant.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")

            if transaction_type_filter == "debit":
                aggregations[f"{merchant_key}_debit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                {"term": {"merchant_name.keyword": merchant}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            elif transaction_type_filter == "credit":
                aggregations[f"{merchant_key}_credit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                {"term": {"merchant_name.keyword": merchant}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            else:
                # Les deux types pour ce marchand
                aggregations[f"{merchant_key}_debit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                {"term": {"merchant_name.keyword": merchant}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }
                aggregations[f"{merchant_key}_credit"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                {"term": {"merchant_name.keyword": merchant}}
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

    def _add_temporal_aggregations(self, aggregations: Dict[str, Any], context: Dict[str, Any]):
        """Ajoute la segmentation temporelle pour les périodes >= 2 mois"""
        from dateutil.relativedelta import relativedelta
        from datetime import date
        import calendar

        if not context["temporal_segmentation"] or not context["date_range"]:
            return

        date_range = context["date_range"]
        transaction_type_filter = context["transaction_type_filter"]

        # Générer les mois de la période
        current_date = date_range["start"].replace(day=1)  # Premier jour du mois
        end_date = date_range["end"]

        while current_date <= end_date:
            month_key = current_date.strftime("%Y_%m")  # 2024_05
            month_name = current_date.strftime("%B_%Y").lower()  # mai_2024

            # Calculer les bornes du mois
            _, last_day = calendar.monthrange(current_date.year, current_date.month)
            month_end = current_date.replace(day=last_day)

            month_filter = {
                "range": {
                    "date": {
                        "gte": current_date.isoformat(),
                        "lte": min(month_end, end_date).isoformat()
                    }
                }
            }

            if transaction_type_filter == "debit":
                aggregations[f"monthly_debit_{month_key}"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                month_filter
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            elif transaction_type_filter == "credit":
                aggregations[f"monthly_credit_{month_key}"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                month_filter
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            else:
                # Les deux types pour ce mois
                aggregations[f"monthly_debit_{month_key}"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "debit"}},
                                month_filter
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }
                aggregations[f"monthly_credit_{month_key}"] = {
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"transaction_type": "credit"}},
                                month_filter
                            ]
                        }
                    },
                    "aggs": {"sum_amount": {"sum": {"field": "amount_abs"}}}
                }

            # Passer au mois suivant
            current_date = current_date + relativedelta(months=1)

    def _optimize_aggregations(self, aggregations: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les optimisations et limitations sur les agrégations"""

        # Limitation à 50 agrégations max
        if len(aggregations) > 50:
            logger.warning(f"Trop d'agrégations ({len(aggregations)}), limitation à 50")
            # Garder les plus importantes : transaction_count + total_* en priorité
            priority_keys = [k for k in aggregations.keys() if k.startswith(('transaction_count', 'total_'))]
            other_keys = [k for k in aggregations.keys() if not k.startswith(('transaction_count', 'total_'))]

            # Garder priorités + compléter jusqu'à 50
            keep_keys = priority_keys + other_keys[:50-len(priority_keys)]
            aggregations = {k: aggregations[k] for k in keep_keys}

        return aggregations

    def _parse_python_strings(self, obj: Any) -> Any:
        """Parse récursivement les strings Python (listes/dicts) en objets natifs"""
        import ast

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                parsed_v = self._parse_python_strings(v)
                if k == "merchant_name" and isinstance(v, str) and v != parsed_v:
                    logger.info(f"✅ Parsed merchant_name: '{v}' -> {parsed_v}")
                result[k] = parsed_v
            return result
        elif isinstance(obj, list):
            return [self._parse_python_strings(item) for item in obj]
        elif isinstance(obj, str):
            # Tenter de parser les strings qui ressemblent à des listes ou dicts Python
            v = obj.strip()
            if (v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}')):
                try:
                    parsed = ast.literal_eval(v)
                    logger.info(f"🔄 Parsed Python string: {v[:80]}... -> {type(parsed).__name__} with {len(parsed) if isinstance(parsed, (list, dict)) else '?'} items")
                    return parsed
                except (ValueError, SyntaxError) as e:
                    # Si échec de parsing, retourner la string telle quelle
                    logger.warning(f"❌ Could not parse Python string: {v[:80]}..., error: {e}")
                    return obj
            return obj
        else:
            return obj

    def _remove_null_values(self, obj: Any) -> Any:
        """Supprime les valeurs null récursivement"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Exclure les valeurs null ET les chaînes "None" (de Jinja2)
                # MAIS garder les objets sérialisés qui ressemblent à des dicts Python
                if v is not None and v != "None" and str(v).strip() != "":
                    # Cas spécial : si c'est une string qui ressemble à un dict Python, la parser
                    if isinstance(v, str) and (v.startswith("{'") and v.endswith("}")):
                        try:
                            # Convertir le dict Python en dict JSON
                            import ast
                            parsed_dict = ast.literal_eval(v)
                            # Si c'est un objet montant, le transformer en filtre Elasticsearch
                            if isinstance(parsed_dict, dict) and "operator" in parsed_dict:
                                cleaned_value = self._to_elasticsearch_amount_filter(parsed_dict)
                            # Si c'est déjà un objet transformé amount (ex: {"gt": 500}), le garder tel quel
                            elif isinstance(parsed_dict, dict) and any(op in parsed_dict for op in ["gt", "gte", "lt", "lte", "eq"]):
                                cleaned_value = parsed_dict
                            else:
                                # Si c'est déjà un objet transformé (ex: {"gt": 500}), le garder tel quel
                                cleaned_value = self._remove_null_values(parsed_dict)
                        except (ValueError, SyntaxError):
                            # Si échec de parsing, garder comme string
                            cleaned_value = self._remove_null_values(v)
                    # Cas spécial : si c'est une string qui ressemble à une liste Python, la parser
                    elif isinstance(v, str) and ((v.startswith("['") and v.endswith("']")) or (v.startswith("[\"") and v.endswith("\"]"))):
                        try:
                            # Convertir la liste Python en vraie liste
                            import ast
                            parsed_list = ast.literal_eval(v)
                            if isinstance(parsed_list, list):
                                logger.debug(f"Parsed Python list string: {v} -> {parsed_list}")
                                cleaned_value = self._remove_null_values(parsed_list)
                            else:
                                cleaned_value = self._remove_null_values(v)
                        except (ValueError, SyntaxError) as e:
                            # Si échec de parsing, garder comme string
                            logger.warning(f"Failed to parse Python list string: {v}, error: {e}")
                            cleaned_value = self._remove_null_values(v)
                    else:
                        cleaned_value = self._remove_null_values(v)
                    if cleaned_value is not None and cleaned_value != "None":
                        result[k] = cleaned_value
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                if item is not None and item != "None":
                    cleaned_item = self._remove_null_values(item)
                    if cleaned_item is not None and cleaned_item != "None":
                        result.append(cleaned_item)
            return result
        else:
            # Convertir "None" en None réel
            if obj == "None":
                return None
            return obj

    def _remove_empty_objects(self, obj: Any) -> Any:
        """Supprime les objets vides récursivement"""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_v = self._remove_empty_objects(v)
                if cleaned_v or cleaned_v == 0 or cleaned_v is False:  # Garder 0 et False
                    cleaned[k] = cleaned_v
            return cleaned if cleaned else None
        elif isinstance(obj, list):
            return [self._remove_empty_objects(item) for item in obj if item]
        else:
            return obj

    def _restore_parameter_types(self, template: CompiledTemplate, query: Dict[str, Any]) -> Dict[str, Any]:
        """Restaure les types corrects des paramètres après rendu Jinja2"""
        # Créer un mapping des paramètres et leurs types
        type_mappings = {}
        for param_name, param_config in template.parameter_mappings.items():
            param_type = param_config.get("type", "string")
            type_mappings[param_name] = param_type
        
        return self._apply_type_mappings(query, type_mappings)
    
    def _apply_type_mappings(self, obj: Any, type_mappings: Dict[str, str], current_path: str = "") -> Any:
        """Applique les mappings de type récursivement"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # Si on a un type mapping direct pour cette clé
                if key in type_mappings:
                    result[key] = self._cast_parameter_type(value, type_mappings[key])
                else:
                    result[key] = self._apply_type_mappings(value, type_mappings, new_path)
            return result
        elif isinstance(obj, list):
            return [self._apply_type_mappings(item, type_mappings, current_path) for item in obj]
        else:
            return obj

    def _validate_template_structure(self, template_data: Dict[str, Any]) -> List[str]:
        """Valide la structure d'un template"""
        errors = []
        
        required_fields = ["template_name", "description", "target_intention", "parameters"]
        for field in required_fields:
            if field not in template_data:
                errors.append(f"Missing required field: {field}")
        
        # Valider les parameter_mappings si présents
        if "parameter_mappings" in template_data:
            for param_name, param_config in template_data["parameter_mappings"].items():
                if not isinstance(param_config, dict):
                    errors.append(f"Parameter mapping for {param_name} must be a dictionary")
                elif "source" not in param_config:
                    errors.append(f"Parameter mapping for {param_name} missing 'source' field")
        
        return errors

    def _parse_cache_duration(self, duration_str: str) -> timedelta:
        """Parse une durée de cache (ex: '1h', '30m', '5s')"""
        match = re.match(r"(\d+)([smhd])", duration_str.lower())
        if not match:
            return timedelta(hours=1)  # Défaut
            
        amount = int(match.group(1))
        unit = match.group(2)
        
        if unit == 's':
            return timedelta(seconds=amount)
        elif unit == 'm':
            return timedelta(minutes=amount)
        elif unit == 'h':
            return timedelta(hours=amount)
        elif unit == 'd':
            return timedelta(days=amount)
        else:
            return timedelta(hours=1)

    def _is_template_cache_valid(self, template: CompiledTemplate) -> bool:
        """Vérifie si un template en cache est encore valide"""
        return datetime.now() - template.compiled_at < template.cache_duration

    async def _precompile_all_templates(self):
        """Précompile tous les templates au démarrage"""
        for template_file in self.templates_dir.rglob("*.json"):
            result = await self.compile_template(template_file)
            if result.success and result.template:
                # Déterminer la clé depuis le target_intention
                target_intention = result.template.template_data.get("target_intention", "")
                if "." in target_intention:
                    self.compiled_templates[target_intention] = result.template
                else:
                    logger.warning(f"Template {result.template.name} has invalid target_intention: {target_intention}")
            else:
                logger.error(f"Failed to precompile template {template_file}: {result.error_message}")

    async def _compile_template_by_intent(self, intent_group: str, intent_subtype: str) -> Optional[CompiledTemplate]:
        """Compile un template par intention"""
        # Chercher le fichier template correspondant
        group_dir = self.templates_dir / intent_group.lower()
        
        possible_files = [
            group_dir / f"{intent_subtype}.json",
            group_dir / f"by_{intent_subtype}.json",
            group_dir / f"{intent_subtype}_search.json"
        ]
        
        # Log pour debugging
        logger.info(f"Recherche template pour {intent_group}.{intent_subtype}")
        logger.info(f"Répertoire de base: {self.templates_dir}")
        logger.info(f"Répertoire groupe: {group_dir}")
        logger.info(f"Fichiers possibles: {[str(f) for f in possible_files]}")
        
        for template_file in possible_files:
            logger.info(f"Test existence: {template_file} -> {template_file.exists()}")
            if template_file.exists():
                result = await self.compile_template(template_file)
                if result.success and result.template:
                    template_key = f"{intent_group}.{intent_subtype}"
                    self.compiled_templates[template_key] = result.template
                    return result.template
                else:
                    logger.error(f"Failed to compile template {template_file}: {result.error_message}")
        
        logger.warning(f"No template found for intent {intent_group}.{intent_subtype}")
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "total_compilations": self.cache_stats["compilations"],
            "templates_in_cache": len(self.compiled_templates)
        }

    async def reload_templates(self):
        """Recharge tous les templates (hot-reload)"""
        logger.info("Reloading all templates...")
        self.compiled_templates.clear()
        await self._precompile_all_templates()
        logger.info(f"Reloaded {len(self.compiled_templates)} templates")
    
    def _to_elasticsearch_amount_filter(self, amount_obj: Any) -> Dict[str, Any]:
        """Convertit un objet montant en filtre Elasticsearch"""
        
        if not amount_obj or not isinstance(amount_obj, dict):
            return None
        
        operator = amount_obj.get('operator', 'eq')
        
        if operator == 'gte':
            return {"gte": amount_obj.get('amount')}
        elif operator == 'gt':
            return {"gt": amount_obj.get('amount')}
        elif operator == 'lte':
            return {"lte": amount_obj.get('amount')}
        elif operator == 'lt':
            return {"lt": amount_obj.get('amount')}
        elif operator == 'eq':
            return {"eq": amount_obj.get('amount')}
        elif operator == 'range':
            return {
                "gte": amount_obj.get('min'),
                "lte": amount_obj.get('max')
            }
        else:
            return {"eq": amount_obj.get('amount', 0)}
    
    def _is_defined_and_not_none(self, value: Any) -> bool:
        """Vérifie si une valeur est définie et non None"""
        return value is not None and value != "None" and str(value).strip() != ""

# Instance globale
_template_engine: Optional[TemplateEngine] = None

async def get_template_engine() -> TemplateEngine:
    """Retourne l'instance globale du moteur de templates"""
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
        await _template_engine.initialize()
    return _template_engine