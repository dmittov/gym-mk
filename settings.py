from dataclasses import dataclass


@dataclass
class Configuration:
    device: str = "cpu"
