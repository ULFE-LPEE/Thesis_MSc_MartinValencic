import numpy as np
import powerfactory as pf
from enum import Enum
class Busbar():
    def __init__(self, data_object: pf.DataObject, name: str, substation: str, id: int):
        self.data_object = data_object
        # Basic busbar data
        self.id = id
        self.name = name
        self.substation = substation

class SynchronousGenerator():
    def __init__(self, data_object: pf.DataObject, name, zone, connected_busbar, type_name, S_rat, U_rat, r_a, x_as, x_d1, base_mva: float = 1.0):
        self.data_object = data_object  # PowerFactory data object, if available
        # Basic element data
        self.name = name                            # Unique name provided by PF
        self.busbar_from = None                     # Connected busbar from
        self.busbar_to = connected_busbar           # Connected busbar to
        self.type = type_name                       # Type of the element
        
        # Type Basic data
        self.S_rat = S_rat                                              # Rated Power           (MVA)
        self.U_rat = U_rat                                              # Rated Voltage         (kV)
        self.zone = zone                                                # Zone                  (str)
        
        # Type RMS parameters
        self.r_a = r_a                                  # Armature Resistance   R_a [p.u]
        self.x_as = x_as                                # Leakage Reactance     X_l [p.u]
        self.x_d1 = x_d1                                # Transient Reactance   X_d' [p.u]

        # Base values
        self.base_mva = base_mva
        self.Z_base = self.U_rat ** 2 / self.base_mva
        self.Y_base = 1 / self.Z_base

        # Calculated parameters
        self.Y_units = self.calculate_admittance()  # [S] Admittance

        # Calculate p.u. value of the admittance
        self.Y = self.Y_units / self.Y_base         # [p.u] Admittance on System base

    def calculate_impedance(self):
        """
        Calculate the impedance of the generator in units values
        """
        # Conver to normal units values
        Z_pu_gen = self.r_a + 1j*(self.x_as + self.x_d1)    # [p.u] impedance on Generator base
        # Z_pu_gen = self.data_object.GetAttribute("typ_id").GetAttribute("rstr") + 1j*(self.data_object.GetAttribute("typ_id").GetAttribute("xstr"))    # [p.u] impedance on Generator base
        Z_ohm = Z_pu_gen * (self.U_rat ** 2) / self.S_rat     # [Ohm] impedance

        return Z_ohm

    def calculate_admittance(self):
        """
        Calculate the admittance of the generator in units values
        """
        Y_sg = 1 / self.calculate_impedance()
        return Y_sg
    
class Line():
    def __init__(self, data_object: pf.DataObject, name, connected_busbar_from, connected_busbar_to, type_name, rated_voltage, length, resistance, reactance, susceptance_effective, susceptance_ground, parallel, base_mva=1.0):
        self.data_object = data_object  # PowerFactory data object, if available
        # Basic element data
        self.name = name                            # Unique name provided by PF
        self.busbar_from = connected_busbar_from    # Connected busbar from
        self.busbar_to = connected_busbar_to        # Connected busbar to
        self.type = type_name                       # Type of the element
        
        # Element basic data
        self.length = length
        self.resistance = resistance                        # [Ohm]
        self.reactance = reactance                          # [Ohm]
        self.susceptance_effective = susceptance_effective  # [uS]
        self.susceptance_ground = susceptance_ground        # [uS]
        self.parallel = parallel

        # Type Basic data
        self.rated_voltage = rated_voltage
        
        # Base MVA for calculations
        self.base_mva = base_mva        # Calculated parameters
        self.Y_line = self.calculate_admittance(self.base_mva)
        self.Y_shunt = self.calculate_shunt_admittance(self.base_mva)

    def calculate_impedance(self, base_mva: float = 1.0):
        """
        Calculate the impedance of the line on the specified base MVA.
        """
        r = self.resistance / (self.rated_voltage ** 2 / base_mva) # Calculate per unit resistance
        x = self.reactance / (self.rated_voltage ** 2 / base_mva) # Calculate per unit reactance
        return (r + 1j * x)
    
    def calculate_admittance(self, base_mva: float = 1.0):
        """
        Calculate the admittance of the line on the specified base MVA.
        """
        Z = self.calculate_impedance(base_mva)
        if Z == 0:
            Z = 1e-12
        Y = 1 / Z
        return Y
    
    def calculate_shunt_admittance(self, base_mva: float = 1.0):
        """
        Calculate the shunt admittance of the line on the specified base MVA.
        """
        b_effective = (self.susceptance_effective * 10**-6) / (base_mva / self.rated_voltage ** 2)
        b_ground = (self.susceptance_ground * 10**-6) / (base_mva / self.rated_voltage ** 2)
        return 1j * (b_effective + b_ground)

class Switch():
    def __init__(self, data_object: pf.DataObject, name, connected_busbar_from, connected_busbar_to, type_name, on_resistance, voltage_level, base_mva: float = 1.0):
        self.data_object = data_object  # PowerFactory data object, if available
        # Basic element data
        self.name = name                            # Unique name provided by PF
        self.busbar_from = connected_busbar_from    # Connected busbar from
        self.busbar_to = connected_busbar_to        # Connected busbar to
        self.type = type_name                       # Type of the element
        
        self.voltage_level = voltage_level
        self.on_resistance = on_resistance # [Ohm]

        # Base values
        self.S_base = base_mva
        self.Z_base = self.voltage_level ** 2 / self.S_base
        self.Y_base = 1 / self.Z_base
        
        # Calculated parameters
        self.Y_units = self.calculate_admittance()

        # Calculate p.u. value of the admittance
        self.Y = self.Y_units / self.Y_base
    
    def calculate_impedance(self):
        # Calculate the impedance of the switch
        Z = self.on_resistance
        return Z
    
    def calculate_admittance(self):
        # Calculate the admittance of the switch
        Y = 1 / self.calculate_impedance()
        return Y
    
class TwoWindingTransformer():
    """
    Represents a two-winding transformer in a power system network.

    Attributes:
        name (str): Name of the transformer.
        bus_from (str): Identifier of the primary (from) bus.
        bus_to (str): Identifier of the secondary (to) bus.
        type_name (str): Type or category of the transformer.
        U_nominal_HV (float): Nominal voltage on the high-voltage side (kV).
        U_nominal_LV (float): Nominal voltage on the low-voltage side (kV).
        S_rat (float): Rated power of the transformer (MVA).
        U_k (float): Percentage short-circuit voltage (%).
        U_k_r (float): Percentage short-circuit resistance voltage (%).
        U_rated_HV (float): Rated voltage on the high-voltage side (kV).
        U_rated_LV (float): Rated voltage on the low-voltage side (kV).
        phase_shift (float): Phase shift angle (radians).
        Y (complex): Transformer admittance on the system base.
        ratio (complex): Off-nominal tap ratio including phase shift.
    """
    def __init__(self, data_object: pf.DataObject, name, bus_from, bus_to, type_name, S_rat, U_k, U_k_r, phase_shift, U_nominal_HV, U_nominal_LV, U_rated_HV, U_rated_LV, parallel, tap_position, voltage_per_tap, tap_side, base_mva: float = 1.0):
        """
        Initializes the TwoWindingTransformer instance with the provided parameters.
        """
        self.data_object = data_object  # PowerFactory data object, if available
        # Basic element data
        self.name = name                            # Unique name provided by PF
        self.busbar_from = bus_from                 # Connected busbar from
        self.busbar_to = bus_to                     # Connected busbar to
        self.type = type_name                       # Type of the element
        
        # Connected busbars voltages
        self.U_nominal_HV = U_nominal_HV    # U_nominal_HV [kV] Nominal voltage HV side
        self.U_nominal_LV = U_nominal_LV    # U_nominal_LV [kV] Nominal voltage LV side
        self.parallel = parallel            # Parallel transformers        # Type parameters
        self.S_rat = S_rat                  # S_rat [MVA] Rated power
        self.U_k = U_k                      # U_k [%] Percentage short-circuit voltage
        self.U_k_r = U_k_r                  # U_k_r [%] Percentage short-circuit resistance voltage
        self.U_rated_HV = U_rated_HV        # U_rated_HV [kV] Rated voltage HV side
        self.U_rated_LV = U_rated_LV        # U_rated_LV [kV] Rated voltage LV side
        self.phase_shift = phase_shift      # Phase shift angle [rad]
        self.tap_position = tap_position    # Tap position
        self.voltage_per_tap = voltage_per_tap # Voltage per tap
        self.tap_side = tap_side            # Tap side ("HV" or "LV")

        # Base MVA for calculations
        self.base_mva = base_mva

        # Calculated parameters
        self.Y = self.calculate_admittance(self.base_mva)
        self.ratio = self.calculate_off_nominal_tap_ratio()

        # Get also PF admittnance for comparison
        err, z_real, z_imag = self.data_object.GetImpedance(self.U_rated_HV, 0)
        if err != 0:
            raise ValueError(f"Error calculating impedance for transformer {self.name}: {err}")
        pf_impedance = complex(z_real, z_imag)
        pf_admittance = 1 / pf_impedance if pf_impedance != 0 else 1e-12

        self.Y_pf = pf_admittance * 100  # Admittance in p.u. on the transformer's base
        self.Y_pf *= ( self.S_rat / base_mva)  # Convert to system base MVA

        # print(f"Transformer {self.name} ---------------------------")
        # print(f"Our admittance                : {self.Y:.4f}")
        # print(f"PF  admittance                : {self.Y_pf:.4f}")
    
    def calculate_impedance(self, base_mva: float = 1.0) -> complex:
        """
        Calculates the transformer's impedance on the specified base MVA.

        Args:
            base_mva (float, optional): System base power in MVA. Defaults to 1.0.

        Returns:
            complex: The transformer's impedance (R + jX) in per-unit on the system base.

        Raises:
            ValueError: If the calculated reactance is not a real number due to invalid inputs.
        """
        # Step 1: Convert percentage values to per-unit on the transformer's base
        Z_pu = self.U_k / 100.0    # Convert U_k from % to per-unit on the transformer's base
        R_pu = self.U_k_r / 100.0  # Convert U_k_r from % to per-unit on the transformer's base

        # Step 2: Calculate reactance on the transformer's base
        Z_pu_squared = Z_pu ** 2
        R_pu_squared = R_pu ** 2
        if Z_pu_squared < R_pu_squared:
            raise ValueError("Invalid transformer parameters: Z_pu squared is less than R_pu squared.")

        X_pu = (Z_pu_squared - R_pu_squared) ** 0.5 # Transformer base impedance

        # Step 3: Adjust per-unit values to the specified base MVA if necessary
        if self.S_rat != base_mva:
            scaling_factor = (base_mva / self.S_rat)
            R_pu *= scaling_factor
            X_pu *= scaling_factor

        # Step 4: Return the impedance as a complex number
        impedance = complex(R_pu, X_pu)
        return impedance
    
    def calculate_admittance(self, base_mva: float = 1.0) -> complex:
        """
        Calculates the transformer's admittance on the specified base MVA.

        Args:
            base_mva (float, optional): System base power in MVA. Defaults to 1.0.

        Returns:
            complex: The transformer's admittance (G + jB) in per-unit on the system base.

        Raises:
            ZeroDivisionError: If the calculated impedance is zero.
        """
        impedance = self.calculate_impedance(base_mva)
        if impedance == 0:
            # Adjust to very small value to avoid division by zero
            impedance = 1e-12

        admittance = 1 / impedance
        return admittance
    
    def calculate_off_nominal_tap_ratio(self) -> complex:
        """
        Calculates the transformer's off-nominal tap ratio, including any phase shift.

        Returns:
            complex: The off-nominal tap ratio as a complex number (magnitude and phase).
        """
        # Calculate the transformer voltage ratio (rated voltages)
        a_tr = self.U_rated_HV / self.U_rated_LV 

        # Calculate the system voltage ratio (nominal voltages)
        a_sys = (self.U_nominal_HV / self.U_nominal_LV)

        # Calculate the off-nominal tap ratio magnitude
        tap_ratio_magnitude = a_tr / a_sys

        # return tap_ratio_magnitude
    
        # Tap changer effect # TODO: Check if this is correctly applied
        t = 1 + self.tap_position * self.voltage_per_tap / 100.0
        tap_ratio = tap_ratio_magnitude * t * np.exp(1j * self.phase_shift)

        return tap_ratio
    
    def get_admittance_matrix_elements(self, base_mva: float = 1.0):
        """
        Calculates the admittance matrix elements for incorporation into the network admittance matrix,
        applying the tap ratio to the correct side as in pandapower.
        """
        Y = self.calculate_admittance(base_mva)
        # Y = self.Y_pf #! TESTING
        a = self.calculate_off_nominal_tap_ratio()
        a_conj = np.conj(a)
        abs_a_squared = abs(a) ** 2

        Y_aa = Y / abs_a_squared
        Y_bb = Y
        Y_ab = - Y / a_conj
        Y_ba = - Y / a

        if self.tap_side == "HV":
            # Tap on HV ("from") side
            Y_aa = Y / abs_a_squared
            Y_bb = Y
            Y_ab = -Y / a_conj
            Y_ba = -Y / a
        elif self.tap_side == "LV":
            # Tap on LV ("to") side
            Y_aa = Y
            Y_bb = Y / abs_a_squared
            Y_ab = -Y / a
            Y_ba = -Y / a_conj
        else:
            raise ValueError("tap_side must be 'HV' or 'LV'")

        return {
            'Y_aa': Y_aa,
            'Y_bb': Y_bb,
            'Y_ab': Y_ab,
            'Y_ba': Y_ba
        }
    
class ThreeWindingTransformer():
    def __init__(
        self,
        data_object: pf.DataObject,
        name,
        u_k_percent_AB,
        u_k_percent_BC,
        u_k_percent_CA,
        S_nom_AB,
        S_nom_BC,
        S_nom_CA,
        bus_HV,
        bus_MV,
        bus_LV,
        base_mva: float = 1.0
    ): 
        self.data_object = data_object
        self.name = name
        self.busbar_from = None  # Three-winding transformers connect to 3 buses
        self.busbar_to = None
        self.type = "ThreeWindingTransformer"
        
        # Type Basic data
        self.u_k_percent_AB = u_k_percent_AB
        self.u_k_percent_BC = u_k_percent_BC
        self.u_k_percent_CA = u_k_percent_CA

        self.S_nom_AB = S_nom_AB
        self.S_nom_BC = S_nom_BC
        self.S_nom_CA = S_nom_CA

        self.name = name
        self.bus_HV = bus_HV
        self.bus_MV = bus_MV
        self.bus_LV = bus_LV
        
        # Base MVA support
        self.base_mva = base_mva

    def Z_ab(self):
        """Per-unit impedance on a 1 MVA base, for the AB side."""
        # Convert S_nom_AB to MVA if given in VA
        S_nom_AB_MVA = self.S_nom_AB
        return (self.u_k_percent_AB / 100.0) * (1.0 / S_nom_AB_MVA)

    def Z_bc(self):
        """Per-unit impedance on a 1 MVA base, for the BC side."""
        S_nom_BC_MVA = self.S_nom_BC
        return (self.u_k_percent_BC / 100.0) * (1.0 / S_nom_BC_MVA)

    def Z_ca(self):
        """Per-unit impedance on a 1 MVA base, for the CA side."""
        S_nom_CA_MVA = self.S_nom_CA
        return (self.u_k_percent_CA / 100.0) * (1.0 / S_nom_CA_MVA)

    def Y_ab(self):
        Zab_pu = self.Z_ab()
        return 1.0 / Zab_pu if Zab_pu != 0 else np.inf

    def Y_bc(self):
        Zbc_pu = self.Z_bc()
        return 1.0 / Zbc_pu if Zbc_pu != 0 else np.inf

    def Y_ca(self):
        Zca_pu = self.Z_ca()
        return 1.0 / Zca_pu if Zca_pu != 0 else np.inf

    def delta_admittance_matrix(self):
        """
        Returns a 3x3 per-unit admittance matrix (on 1 MVA base)
        for the three nodes A, B, C in a delta connection.

        Off-diagonals: -Y_xy
        Diagonals: sum of admittances to that node.
        """
        Yab = self.Y_ab()
        Ybc = self.Y_bc()
        Yca = self.Y_ca()

        Y_matrix = np.zeros((3, 3), dtype=complex)

        # Off-diagonal
        Y_matrix[0, 1] = -Yab  # A-B
        Y_matrix[1, 0] = -Yab
        Y_matrix[1, 2] = -Ybc  # B-C
        Y_matrix[2, 1] = -Ybc
        Y_matrix[0, 2] = -Yca  # A-C
        Y_matrix[2, 0] = -Yca

        # Diagonal
        Y_matrix[0, 0] = Yab + Yca
        Y_matrix[1, 1] = Yab + Ybc
        Y_matrix[2, 2] = Ybc + Yca

        return Y_matrix

# class ThreeWindingTransformer():
#     def __init__(
#         self,
#         data_object: pf.DataObject,
#         name,
#         bus_HV,
#         bus_MV,
#         bus_LV,
#         U_nominal_HV,
#         U_nominal_MV,
#         U_nominal_LV,
#         base_mva: float = 100.0
#     ): 
#         self.data_object = data_object
#         self.name = name
#         self.busbar_from = None  # Three-winding transformers connect to 3 buses
#         self.busbar_to = None
#         self.type = "ThreeWindingTransformerPF"
        
#         self.bus_HV = bus_HV
#         self.bus_MV = bus_MV
#         self.bus_LV = bus_LV
        
#         # Nominal voltages
#         self.U_nominal_HV = U_nominal_HV
#         self.U_nominal_MV = U_nominal_MV
#         self.U_nominal_LV = U_nominal_LV
        
#         # Base MVA (default 100 MVA)
#         self.base_mva = base_mva

#     def get_pf_impedance(self, voltage_level, winding_pair):
#         """
#         Get impedance from PowerFactory using GetImpedance method.
        
#         Args:
#             voltage_level (float): Reference voltage in kV
#             winding_pair (int): Winding pair identifier (0=HV-MV, 1=HV-LV, 2=MV-LV)
        
#         Returns:
#             complex: Impedance as complex number
#         """
#         err, z_real, z_imag = self.data_object.GetImpedance(voltage_level, winding_pair)
#         if err != 0:
#             raise ValueError(f"Error calculating impedance for transformer {self.name}, winding pair {winding_pair}: {err}")
        
#         return complex(z_real, z_imag)

#     def Y_ab(self):
#         """Admittance between HV and MV windings (A-B)"""
#         Z_ab = self.get_pf_impedance(self.U_nominal_HV, 0)  # HV-MV impedance
#         if Z_ab == 0:
#             Z_ab = 1e-12
#         return 1.0 / Z_ab

#     def Y_bc(self):
#         """Admittance between MV and LV windings (B-C)"""
#         Z_bc = self.get_pf_impedance(self.U_nominal_MV, 2)  # MV-LV impedance
#         if Z_bc == 0:
#             Z_bc = 1e-12
#         return 1.0 / Z_bc

#     def Y_ca(self):
#         """Admittance between LV and HV windings (C-A)"""
#         Z_ca = self.get_pf_impedance(self.U_nominal_HV, 1)  # HV-LV impedance
#         if Z_ca == 0:
#             Z_ca = 1e-12
#         return 1.0 / Z_ca

#     def delta_admittance_matrix(self):
#         """
#         Returns a 3x3 per-unit admittance matrix on the specified system base (default 100 MVA)
#         for the three nodes A (HV), B (MV), C (LV) in a delta connection.
#         Uses PowerFactory's GetImpedance method for impedance calculation.

#         Off-diagonals: -Y_xy
#         Diagonals: sum of admittances to that node.
#         """
#         Yab = self.Y_ab() * self.base_mva  # HV-MV admittance
#         Ybc = self.Y_bc() * self.base_mva  # MV-LV admittance  
#         Yca = self.Y_ca() * self.base_mva  # LV-HV admittance

#         # Convert to system base if needed
#         # PowerFactory returns values on transformer base, convert to system base
#         # This assumes the impedances from PF are already in per-unit on appropriate base
        
#         Y_matrix = np.zeros((3, 3), dtype=complex)

#         # Off-diagonal elements (negative admittances)
#         Y_matrix[0, 1] = -Yab  # HV-MV
#         Y_matrix[1, 0] = -Yab
#         Y_matrix[1, 2] = -Ybc  # MV-LV
#         Y_matrix[2, 1] = -Ybc
#         Y_matrix[0, 2] = -Yca  # HV-LV
#         Y_matrix[2, 0] = -Yca

#         # Diagonal elements (sum of admittances connected to each node)
#         Y_matrix[0, 0] = Yab + Yca  # HV node
#         Y_matrix[1, 1] = Yab + Ybc  # MV node
#         Y_matrix[2, 2] = Ybc + Yca  # LV node

#         return Y_matrix

class ShuntType(Enum):
    R_L_C = 0,
    R_L = 1,
    C = 2,
    R_L_C_Rp = 3,
    R_L_C1_C2_Rp = 4,

class ShuntElement():
    def __init__(self, data_object: pf.DataObject, name, bus_to, U_rat, shunt_type, base_mva: float = 1.0) -> None:
        self.data_object = data_object
        self.name = name
        self.U_rat = U_rat
        self.bus_to = bus_to
        self.shunt_type = shunt_type
        self.base_mva = base_mva
        self.Y = self.calculate_admittance(self.base_mva)

    def calculate_admittance(self, base_mva: float = 1.0):
        if self.shunt_type == ShuntType.R_L_C.value[0]:
            # Layout Parameters (per Step)
            B = self.data_object.GetAttribute("bcap")  # [uS]
            X = self.data_object.GetAttribute("xrea")  # [Ohm]
            R = self.data_object.GetAttribute("rrea")  # [Ohm]
            nr_active_step = self.data_object.GetAttribute("ncapa")  # integer
            # TODO: Update parameters if step is different than 1

            # Calculate conductance
            G = R / (R ** 2 + X ** 2)   # [S]

            # Calculate admittance
            Y = G + 1j * B * 10 ** -6   # [S]

            # Calculate per unit admittance
            Y_base = base_mva / (self.U_rat ** 2)  # [S]
            Y_pu = Y / Y_base

            return Y_pu
        
        elif self.shunt_type == ShuntType.R_L.value[0]:
            X = self.data_object.GetAttribute("xrea")  # [Ohm]
            R = self.data_object.GetAttribute("rrea")

            # Calculate conductance
            G = R / (R ** 2 + X ** 2)

            # Calculate admittance
            Y = G

            # Calculate per unit admittance
            Y_base = base_mva / (self.U_rat ** 2)
            Y_pu = Y / Y_base

            return Y_pu
        
        elif self.shunt_type == ShuntType.C.value[0]:
            B = self.data_object.GetAttribute("bcap")        # [uS]
            G_p = self.data_object.GetAttribute("gparac")    # [uS]

            # Calculate admittance
            Y = 1j * B * 10 ** -6 + G_p * 10 ** -6

            # Calculate per unit admittance
            Y_base = base_mva / (self.U_rat ** 2)
            Y_pu = Y / Y_base

            return Y_pu
        elif self.shunt_type == ShuntType.R_L_C_Rp.value[0]:
            pass
        elif self.shunt_type == ShuntType.R_L_C1_C2_Rp.value[0]:
            # Get resonant frequency
            f_res = self.data_object.GetAttribute("fres")  # [Hz]
            omega = 2 * np.pi * f_res
            C1 = self.data_object.GetAttribute("c1")     # [uF]
            B1 = omega * C1 * 10 ** -6
            C2 = self.data_object.GetAttribute("c2")     # [uF]
            B2 = omega * C2 * 10 ** -6
            X = self.data_object.GetAttribute("xrea")
            R = self.data_object.GetAttribute("rrea")
            R_p = self.data_object.GetAttribute("rpara")

            # Calculate branch admittance
            Z_branch = R + 1j * X - 1j/B1
            Y_branch = 1 / Z_branch

            # Calculate parallel admittance
            Y_p = 1 / R_p
            
            # C2
            Y_c2 = 1j * B2

            # Calculate total admittance
            # Y = Y_branch + Y_p
            Y = (Y_branch + Y_p) * Y_c2 / (Y_branch + Y_p + Y_c2) # TODO

            # Convert to system base
            Y_base = base_mva / (self.U_rat ** 2)

            return Y / Y_base
        else:
            raise ValueError("Invalid shunt type. Must be 'capacitor' or 'reactor'.")

class Load():
    def __init__(self, data_object: pf.DataObject, name, connected_busbar, type_name, P, Q, voltage, base_mva: float = 1.0):
        self.data_object = data_object  # PowerFactory data object, if available
        # Basic element data
        self.name = name                            # Unique name provided by PF
        self.busbar_from = None                     # Connected busbar from
        self.busbar_to = connected_busbar           # Connected busbar to
        self.type = type_name                       # Type of the element
        
        self.P = P
        self.Q = Q
        self.voltage = voltage
        self.base_mva = base_mva
        self.Y = self.calculate_admittance(self.base_mva)

    def calculate_power(self, base_mva: float = 1.0):
        return self.P/base_mva - 1j * self.Q/base_mva
    
    def calculate_admittance(self, base_mva: float = 1.0):
        S = self.calculate_power(base_mva)
        voltage = self.voltage
        Y = S / (voltage ** 2)
        return Y
    
class CommonImpedance():
    '''
    This is simplified model of a Two-Winding Transformer. It is used to model the impedance between two buses.
    '''
    def __init__(self, data_object: pf.DataObject, name, bus_from, bus_to, R, X, U_nominal_HV, U_nominal_LV, S_rat, tap_ratio, phase_shift, base_mva: float = 1.0):
        # TODO: If tap-ratio and phase shift are not 1 and 0, respectively, then the impedance should be calculated differently
        self.data_object = data_object
        self.name = name
        self.bus_from = bus_from
        self.bus_to = bus_to
        self.R = R  # [p.u]
        self.X = X  # [p.u]
        self.U_nom_HV = U_nominal_HV  # [kV]
        self.U_nom_LV = U_nominal_LV  # [kV]
        self.S_rat = S_rat  # [MVA]
        self.tap_ratio = tap_ratio
        self.phase_shift = phase_shift

        # Calculate base parameters
        self.base_mva = base_mva
        self.Z_base = (self.U_nom_HV ** 2) / self.base_mva  # [Ohm]
        self.Y_base = 1 / self.Z_base                     # [S]

        # Calculate admittance
        self.Y_units = self.calculate_admittance()

        # Calculate per unit admittance in desired base
        self.Y = self.Y_units / self.Y_base
    
    def calculate_impedance(self):
        # Check if parameters are 0 and assign very small values to them
        self.R = self.R if self.R != 0 else 1e-12
        self.X = self.X if self.X != 0 else 1e-12

        # Convert R and X from element p.u. to normal units
        Z_units = complex(self.R, self.X) * self.S_rat / (self.U_nom_HV ** 2)

        # Return the impedance in normal units
        return Z_units
    
    def calculate_admittance(self):
        Z = self.calculate_impedance()
        Y = 1 / Z
        return Y

class ExternalGrid():
    def __init__(self, data_object: pf.DataObject, name, connected_busbar, U_rat, S_sc, c_factor, r_x_ratio, base_mva: float = 1.0):
        self.data_object = data_object
        self.name = name
        self.busbar_from = None
        self.busbar_to = connected_busbar
        self.type = "ExternalGrid"
        self.U_rat = U_rat
        self.S_sc = S_sc
        self.c_factor = c_factor
        self.R_X = r_x_ratio

        # Calculate base parameters
        self.base_mva = base_mva
        self.S_base = base_mva
        self.Z_base = (self.U_rat ** 2) / self.S_base
        self.Y_base = 1 / self.Z_base

        # Calculate admittance (base values)
        self.Y = self.calculate_admittance() / self.Y_base

    def calculate_impedance(self):
        Z_sc = (self.U_rat ** 2) / self.S_sc
        R_sc = Z_sc * (self.R_X / (1 + self.R_X)) * self.c_factor
        X_sc = Z_sc * (1/np.sqrt(1 + self.R_X**2)) * self.c_factor
        Z_sc_complex = complex(R_sc, X_sc)
        return Z_sc_complex
    
    def calculate_admittance(self):
        Z = self.calculate_impedance()
        return 1 / Z

class VoltageSourceAC():
    def __init__(self, data_object: pf.DataObject, name, connected_busbar, U_rat, R, X, base_mva: float = 1.0):
        self.data_object = data_object
        self.name = name
        self.busbar_from = None
        self.busbar_to = connected_busbar
        self.type = "VoltageSourceAC"
        self.U_rat = U_rat
        self.R = R # [Ohm]
        self.X = X # [Ohm]

        # Calculate base parameters
        self.base_mva = base_mva
        self.S_base = base_mva
        self.Z_base = (self.U_rat ** 2) / self.S_base
        self.Y_base = 1 / self.Z_base

        # Calculate admittance (base values)
        self.Y = self.calculate_admittance() / self.Y_base

    def calculate_impedance(self):
        # Check if parameters are 0 and assign very small values to them
        self.R = self.R if self.R != 0 else 1e-12
        self.X = self.X if self.X != 0 else 1e-12

        Z = complex(self.R, self.X)
        return Z
    
    def calculate_admittance(self):
        Z = self.calculate_impedance()
        return 1 / Z