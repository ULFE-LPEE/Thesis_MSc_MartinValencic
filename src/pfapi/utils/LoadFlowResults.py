from dataclasses import dataclass

@dataclass
class BusLoadFlowResult:
    name: str
    voltage: float  # Voltage magnitude in pu or kV
    angle: float   # Voltage angle in degrees