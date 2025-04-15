import re


def get_flux_controlnet_mode(controlnet_type: str) -> int:
    mapping = {
        "canny": 0,
        "scribble": 0,
        "softedge": 0,
        "tile": 1,
        "depth": 2,
        "blur": 3,
        "pose": 4,
        "gray": 5,
        "low_quality": 6
    }
    key = controlnet_type.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported controlnet type: {controlnet_type}")
    return mapping[key]


def parse_controlnets(raw: str):
    pattern = re.compile(r"(\w+)(?:\(([\d.]+)\))?")
    controlnets = []

    for match in pattern.finditer(raw):
        net_type = match.group(1)
        weight = float(match.group(2)) if match.group(2) else 1.0
        controlnets.append({"type": net_type.strip(), "weight": weight})
    
    return controlnets