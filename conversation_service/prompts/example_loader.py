from pathlib import Path
import yaml


def load_yaml_examples(filename: str, header: str) -> str:
    """Load YAML examples and format them for inclusion in prompts."""
    path = Path(__file__).with_name(filename)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    lines = [header]
    for idx, ex in enumerate(data, 1):
        lines.append(f"Exemple {idx} - {ex['description']} :")
        lines.append(f"ENTRÉE: {ex['input']}")
        lines.append("SORTIE:")
        lines.append(ex['output'])
        lines.append("")
    return "\n".join(lines).strip()
