import powerfactory as pf
from typing import List, Dict
from .LoadFlowResults import BusLoadFlowResult
import cmath

from .Elements import Busbar, Line, SynchronousGenerator, Load, TwoWindingTransformer, Switch, ThreeWindingTransformer, ShuntElement, ExternalGrid, VoltageSourceAC, CommonImpedance

class Converter:
    @staticmethod
    def convert_busbars(busbars: List[pf.DataObject]):
        converted_busbars: List[Busbar] = []

        # Convert PowerFactory busbars to Busbar objects
        for id, busbar in enumerate(busbars):
            new_busbar = Busbar(
                data_object=busbar,
                name=busbar.GetAttribute('loc_name'),
                id=id,
                substation=busbar.GetAttribute('cpSubstat'),
            )
            converted_busbars.append(new_busbar)

        return converted_busbars

    @staticmethod
    def convert_lines(pf_lines: List[pf.DataObject], busbar_name_to_index: object, base_mva: float = 1.0) -> List[Line]:
        """
        Function converts PowerFactory ElmLne objects to Line objects
        """
        lines: List[Line] = []

        for line in pf_lines:
            # Get line type data
            line_type: pf.DataObject = line.GetAttribute('typ_id')
            if not line_type:
                # TODO: IDK yet what to do with this
                continue
            type_name = line_type.GetAttribute('loc_name')
            rated_voltage = line_type.GetAttribute('uline')
            r_per_km = line_type.GetAttribute('rline')
            x_per_km = line_type.GetAttribute('xline')

            # Line type Load-Flow data
            b_per_km = line_type.GetAttribute('bline')
            b0_per_km = line_type.GetAttribute('bline0')

            # Get Line Basic Data
            name = line.GetAttribute('loc_name')
            length = line.GetAttribute('dline')

            cub_from = line.GetAttribute("bus1")
            cub_to = line.GetAttribute("bus2")
            connected_bus_from = line.GetAttribute("r:bus1:e:cBusBar")  
            connected_bus_to = line.GetAttribute("r:bus2:e:cBusBar")

            connections = line.GetConnectedElements(1, 1, 1)
            if len(connections) != 2:
                print("[WARNING] Line has none more than 2 connection --> Line: ", line.GetAttribute('loc_name'))
                continue

            bus_from_name = connected_bus_from.GetAttribute("loc_name")
            bus_to_name = connected_bus_to.GetAttribute("loc_name")

            cub_from_connection = cub_from.IsConnected(connected_bus_from, 1)
            cub_to_connection = cub_to.IsConnected(connected_bus_to, 1)
            if (cub_from_connection == 0 or cub_to_connection == 0):
                print("[WARNING] Line has less than 2 active conections. Will not include in calculations--> Line: ", line.GetAttribute('loc_name'))
                continue

            # IDK if this is okay yet
            if bus_from_name not in busbar_name_to_index or bus_to_name not in busbar_name_to_index:
                print("[WARNING] Busbar not in busbar_name_to_index --> Line: ", line.GetAttribute('loc_name'))
                print(" Will not include this line! \n")
                continue

            # Get Line Resulting Values (already calculated values by PF)
            R = line.GetAttribute("R1")
            X = line.GetAttribute("X1")
            B1 = line.GetAttribute("B1")
            B0 = line.GetAttribute("B0")
            # print(line.GetAttributeUnit("R1"))

            parallel = line.GetAttribute("nlnum")

            new_line = Line(
                data_object=line,
                name=name,
                connected_busbar_from=bus_from_name,
                connected_busbar_to=bus_to_name,
                type_name=type_name,
                rated_voltage=rated_voltage,
                length=length,
                resistance=R,
                reactance=X,
                susceptance_effective=B1,
                susceptance_ground=B0,
                parallel=parallel,
                base_mva=base_mva
            )
            lines.append(new_line)
        return lines

    @staticmethod
    def convert_synchronous_machines(machines: List[pf.DataObject], base_mva: float = 1.0) -> List[SynchronousGenerator]:
        generators: List[SynchronousGenerator] = []
        for machine in machines:
            # Get machine data
            machine_type: pf.DataObject = machine.GetAttribute("typ_id")

            connections = machine.GetConnectedElements(1, 1, 1)
            if len(connections) != 1:
                print("[WARNING] SynMachine has none more than one connection --> SynMachine: ", machine.GetAttribute('loc_name'))
                continue

            connected_bus = connections[0]
            bus_name = connected_bus.GetAttribute("loc_name")

            # Get zone
            zone_attr = machine.GetAttribute("cpZone")
            zone = zone_attr.GetAttribute("loc_name") if zone_attr else "None"

            generator = SynchronousGenerator(
                data_object=machine,
                name=machine.GetAttribute('loc_name'),
                zone=zone,
                connected_busbar=bus_name,
                type_name=machine_type.GetAttribute('loc_name'),
                S_rat=machine_type.GetAttribute('sgn'),
                U_rat=machine_type.GetAttribute('ugn'),
                r_a=machine_type.GetAttribute('rstr'),
                x_as=machine_type.GetAttribute('xl'),
                x_d1=machine_type.GetAttribute('xds'), # xds for standard and xstr for classical
                base_mva=base_mva
            )
            generators.append(generator)
        return generators

    @staticmethod
    def convert_two_winding_transformers(pf_transformers: List[pf.DataObject], base_mva: float = 1.0) -> List[TwoWindingTransformer]:
        transformers: List[TwoWindingTransformer] = []
        for transformer in pf_transformers:
            connections = transformer.GetConnectedElements(1, 1, 1)
            if len(connections) != 2:
                print("[WARNING] Trafo has none more than 2 connection --> Trafo: ", transformer.GetAttribute('loc_name'))
                continue
            
            connected_bus_from = transformer.GetAttribute("r:bushv:e:cBusBar")  # HV
            connected_bus_to = transformer.GetAttribute("r:buslv:e:cBusBar")    # LV
            bus_from_name = connected_bus_from.GetAttribute("loc_name")
            bus_to_name = connected_bus_to.GetAttribute("loc_name")

            busbar_HV_voltage = connected_bus_from.GetAttribute("uknom")
            busbar_LV_voltage = connected_bus_to.GetAttribute("uknom")

            # Get 2 Winding transformer type
            trafo_type: pf.DataObject = transformer.GetAttribute("typ_id")
            rated_power = trafo_type.GetAttribute("strn")
            rated_voltage_HV = trafo_type.GetAttribute("utrn_h")
            rated_voltage_LV = trafo_type.GetAttribute("utrn_l")
            u_k = trafo_type.GetAttribute("uktr")
            u_k_r = trafo_type.GetAttribute("uktrr")
            phase_shift = trafo_type.GetAttribute("nt2ag")
            phi_rad = phase_shift * (cmath.pi / 6)

            tap_position = transformer.GetAttribute("nntap")
            voltage_per_tap = transformer.GetAttribute("t:dutap")
            tap_side = trafo_type.GetAttribute("tap_side")
            if tap_side == 1:
                tap_side = "LV"
            else :
                tap_side = "HV"

            new_transformer = TwoWindingTransformer(
                data_object=transformer,
                name=transformer.GetAttribute('loc_name'),
                bus_from=bus_from_name,
                bus_to=bus_to_name,
                type_name=trafo_type.GetAttribute('loc_name'),
                S_rat=rated_power,
                U_k=u_k,
                U_k_r=u_k_r,
                phase_shift=phi_rad,                U_nominal_HV=busbar_HV_voltage,
                U_nominal_LV=busbar_LV_voltage,
                U_rated_HV=rated_voltage_HV,
                U_rated_LV=rated_voltage_LV,
                parallel=transformer.GetAttribute("ntnum"),
                tap_position=tap_position,
                voltage_per_tap=voltage_per_tap,
                tap_side=tap_side,
                base_mva=base_mva
            )
            transformers.append(new_transformer)
        return transformers

    @staticmethod
    def convert_loads(pf_loads: List[pf.DataObject], load_flow_results: Dict[str, BusLoadFlowResult], base_mva: float = 1.0) -> List[Load]:
        loads: List[Load] = []
        for load in pf_loads:
            connections = load.GetConnectedElements(1, 1, 1)
            if len(connections) != 1:
                print("[WARNING] Load has none more than one connection --> Load: ", load.GetAttribute('loc_name'))
                continue

            connected_bus = connections[0]
            bus_name = connected_bus.GetAttribute("loc_name")
            P_init = load.GetAttribute("plini")
            Q_init = load.GetAttribute("qlini")
            scaling_factor = load.GetAttribute("scale0")
            P_init = P_init * scaling_factor
            Q_init = Q_init * scaling_factor
            
            new_load = Load(
                data_object=load,
                name=load.GetAttribute('loc_name'),
                connected_busbar=bus_name,
                type_name=None,
                P=P_init,
                Q=Q_init,
                voltage=load_flow_results[bus_name].voltage,
                base_mva=base_mva
            )
            loads.append(new_load)
        return loads

    @staticmethod
    def convert_switches(switches: List[pf.DataObject], base_mva: float = 1.0) -> List[Switch]:
        switchs: List[Switch] = []
        for switch in switches:
            is_closed = switch.IsClosed() # type: ignore # 1 = closed, 0 = open
            if (is_closed == 0):
                continue

            cub_from = switch.GetAttribute("bus1")
            cub_to = switch.GetAttribute("bus2")

            connected_bus_from = switch.GetAttribute("r:bus1:e:cBusBar")  
            connected_bus_to = switch.GetAttribute("r:bus2:e:cBusBar")

            connections = switch.GetConnectedElements(1, 1, 1)
            if len(connections) != 2:
                print("[WARNING] Switch has none more than 2 connection --> Switch: ", switch.GetAttribute('loc_name'))
                continue

            cub_from_connection = cub_from.IsConnected(connected_bus_from, 1)
            cub_to_connection = cub_to.IsConnected(connected_bus_to, 1)
            if (cub_from_connection == 0 or cub_to_connection == 0):
                print("[WARNING] Switch has less than 2 active conections. Will not include in calculations--> Switch: ", switch.GetAttribute('loc_name'))
                continue

            bus_from_name = connected_bus_from.GetAttribute("loc_name")
            bus_to_name = connected_bus_to.GetAttribute("loc_name")

            voltage_level = connected_bus_from.GetAttribute("uknom")

            # Generate unique ID
            substation = switch.GetAttribute('cpSubstat')
            substation_name = substation.GetAttribute('loc_name') if substation is not None else "None"
            element_id = Converter.generate_unique_name(switch.GetAttribute("loc_name"), substation_name)

            # Get switch type data if there is any otherwise set default value
            try:
                swith_type: pf.DataObject = switch.GetAttribute('typ_id')
                on_resistance = swith_type.GetAttribute("R_on")
            except:
                on_resistance = 1e-6

            new_switch = Switch(
                data_object=switch,
                name=element_id,
                connected_busbar_from=bus_from_name,
                connected_busbar_to=bus_to_name,
                type_name=switch.GetAttribute('typ_id'),
                on_resistance=on_resistance,
                voltage_level=voltage_level,
                base_mva=base_mva
            )
            switchs.append(new_switch)
        return switchs
    
    @staticmethod
    def convert_three_winding_transformers(pf_transformers: List[pf.DataObject], base_mva: float = 1.0) -> List[ThreeWindingTransformer]:
        transformers: List[ThreeWindingTransformer] = []

        for transformer in pf_transformers:
            connections = transformer.GetConnectedElements(1, 1, 1)
            if len(connections) != 3:
                print("[WARNING] 3WindTR has none more than 2 connection --> 3WindTR: ", transformer.GetAttribute('loc_name'))
                continue

            connected_bus_HV = transformer.GetAttribute("r:bushv:e:cBusBar")
            connected_bus_MV = transformer.GetAttribute("r:busmv:e:cBusBar")
            connected_bus_LV = transformer.GetAttribute("r:buslv:e:cBusBar")

            bus_LV_name = connected_bus_LV.GetAttribute("loc_name")
            bus_MV_name = connected_bus_MV.GetAttribute("loc_name")
            bus_HV_name = connected_bus_HV.GetAttribute("loc_name")

            bus_LV_voltage = connected_bus_LV.GetAttribute("uknom")
            bus_MV_voltage = connected_bus_MV.GetAttribute("uknom")
            bus_HV_voltage = connected_bus_HV.GetAttribute("uknom")

            # Get 3 Winding transformer type
            trafo_type: pf.DataObject = transformer.GetAttribute("typ_id")

            # Get Rated Power
            rated_power_HV = trafo_type.GetAttribute("strn3_h")
            rated_power_MV = trafo_type.GetAttribute("strn3_m")
            rated_power_LV = trafo_type.GetAttribute("strn3_l")

            # Get Rated Voltage
            rated_voltage_HV = trafo_type.GetAttribute("utrn3_h")
            rated_voltage_MV = trafo_type.GetAttribute("utrn3_m")
            rated_voltage_LV = trafo_type.GetAttribute("utrn3_l")

            # Get U_k and U_k_r
            u_k_HV_to_MV = trafo_type.GetAttribute("uktr3_h")
            u_k_MV_to_LV = trafo_type.GetAttribute("uktr3_m")
            u_k_HV_to_LV = trafo_type.GetAttribute("uktr3_l")

            u_k_r_HV_to_MV = trafo_type.GetAttribute("uktrr3_h")
            u_k_r_MV_to_LV = trafo_type.GetAttribute("uktrr3_m")
            u_k_r_HV_to_LV = trafo_type.GetAttribute("uktrr3_l")            # Get Phase Shift
            phase_shift_HV = trafo_type.GetAttribute("nt3ag_h")
            phase_shift_MV = trafo_type.GetAttribute("nt3ag_m")
            phase_shift_LV = trafo_type.GetAttribute("nt3ag_l")

            phi_rad_HV = phase_shift_HV * (cmath.pi / 6)
            phi_rad_MV = phase_shift_MV * (cmath.pi / 6)
            phi_rad_LV = phase_shift_LV * (cmath.pi / 6)

            new_transformer = ThreeWindingTransformer(
                data_object=transformer,
                name=transformer.GetAttribute('loc_name'),
                bus_HV=bus_HV_name,
                bus_MV=bus_MV_name,
                bus_LV=bus_LV_name,
                u_k_percent_AB=u_k_HV_to_MV,
                u_k_percent_BC=u_k_MV_to_LV,
                u_k_percent_CA=u_k_HV_to_LV,
                S_nom_AB=rated_power_HV,
                S_nom_BC=rated_power_MV,
                S_nom_CA=rated_power_LV,
                base_mva=base_mva
            )

            # new_transformer = ThreeWindingTransformer(
            #     data_object=transformer,
            #     name=transformer.GetAttribute('loc_name'),
            #     bus_HV=bus_HV_name,
            #     bus_MV=bus_MV_name,
            #     bus_LV=bus_LV_name,
            #     U_nominal_HV=rated_power_HV,
            #     U_nominal_MV=rated_power_MV,
            #     U_nominal_LV=rated_power_LV,
            #     base_mva=base_mva
            # )
            
            transformers.append(new_transformer)
        return transformers

    @staticmethod
    def convert_shunt_elements(pf_shunt_elements: List[pf.DataObject], base_mva: float = 1.0) -> List[ShuntElement]:
        shunt_elements: List[ShuntElement] = []
        for shunt_element in pf_shunt_elements:
            # Check if shunt element in service
            is_in_service = shunt_element.GetAttribute("outserv")
            if is_in_service == 1:
                continue            # Get shunt element data
            connected_terminal: pf.DataObject = shunt_element.GetAttribute("bus1")
            busbar_name = connected_terminal.GetParent().GetAttribute("loc_name")
            substation = connected_terminal.GetParent().GetAttribute("cpSubstat")
            
            if substation is not None:
                substation_name = substation.GetAttribute("loc_name")
            else:
                substation_name = "None"

            shunt_type = shunt_element.GetAttribute("shtype")
            U_rat = shunt_element.GetAttribute("ushnm")

            new_shunt_element = ShuntElement(
                data_object=shunt_element,
                name=shunt_element.GetAttribute('loc_name'),
                bus_to=busbar_name,
                U_rat=U_rat,
                shunt_type=shunt_type,
                base_mva=base_mva
            )
            shunt_elements.append(new_shunt_element)
        return shunt_elements
    
    @staticmethod
    def convert_external_grids(pf_external_grids: List[pf.DataObject], base_mva: float = 1.0) -> List[ExternalGrid]:
        external_grids: List[ExternalGrid] = []
        for external_grid in pf_external_grids:
            connections = external_grid.GetConnectedElements(1, 1, 1)
            if len(connections) != 1:
                print("[WARNING] ExtGrid has none more than one connection --> ExtGrid: ", external_grid.GetAttribute('loc_name'))
                continue

            connected_bus = connections[0]
            bus_name = connected_bus.GetAttribute("loc_name")
            voltage_level = connected_bus.GetAttribute("uknom")            # Get Load-Flow data
            S_sc = external_grid.GetAttribute("snss")
            c_factor = external_grid.GetAttribute("cmax")
            r_x_ratio = external_grid.GetAttribute("rntxn")

            new_external_grid = ExternalGrid(
                data_object=external_grid,
                name=external_grid.GetAttribute('loc_name'),
                connected_busbar=bus_name,
                U_rat=voltage_level,
                S_sc=S_sc,
                c_factor=c_factor,
                r_x_ratio=r_x_ratio,
                base_mva=base_mva
            )
            external_grids.append(new_external_grid)
        return external_grids
    
    @staticmethod
    def convert_voltage_source_AC(pf_voltage_sources_AC: List[pf.DataObject], base_mva: float = 1.0) -> List[VoltageSourceAC]:
        voltage_sources: List[VoltageSourceAC] = []
        for voltage_source in pf_voltage_sources_AC:
            connections = voltage_source.GetConnectedElements(1, 1, 1)
            if len(connections) != 1:
                print("[WARNING] voltage_source has none more than one connection --> voltage_source: ", voltage_source.GetAttribute('loc_name'))
                continue

            connected_bus = connections[0]
            bus_name = connected_bus.GetAttribute("loc_name")

            U_rat = voltage_source.GetAttribute("Unom")            # Get Load-Flow data
            U_magnitude = voltage_source.GetAttribute("usetp") #?? Pomojem ne rabim
            U_angle = voltage_source.GetAttribute("phisetp") #?? Pomojem ne rabim
            R = voltage_source.GetAttribute("R1")
            X = voltage_source.GetAttribute("X1")

            new_external_grid = VoltageSourceAC(
                data_object=voltage_source,
                name=voltage_source.GetAttribute('loc_name'),
                connected_busbar=bus_name,
                U_rat=U_rat,
                R=R,
                X=X,
                base_mva=base_mva
            )
            voltage_sources.append(new_external_grid)
        return voltage_sources

    @staticmethod
    def convert_common_impedances(pf_common_impedances: List[pf.DataObject], base_mva: float = 1.0) -> List[CommonImpedance]:
        common_impedances: List[CommonImpedance] = []
        for common_impedance in pf_common_impedances:
            # Check if external grid in service
            is_in_service = common_impedance.GetAttribute("outserv")
            if is_in_service == 1:
                continue

            connections = common_impedance.GetConnectedElements(1, 1, 1)
            if len(connections) != 2:
                print("[WARNING] Zpu has none more than 2 connection --> Zpu: ", common_impedance.GetAttribute('loc_name'))
                continue

            connected_bus_from = connections[0]
            connected_bus_to = connections[1]
            bus_from_name = connected_bus_from.GetAttribute("loc_name")
            bus_to_name = connected_bus_to.GetAttribute("loc_name")

            busbar_HV_voltage = connected_bus_from.GetAttribute("uknom")
            busbar_LV_voltage = connected_bus_to.GetAttribute("uknom")
            S_rat = common_impedance.GetAttribute("Sn")      

            # Get Ratio and Shift data
            tap_ratio = common_impedance.GetAttribute("uratio")
            nominal_phase_shift = common_impedance.GetAttribute("nphshift")
            nominal_phase_shift_rd = nominal_phase_shift * (cmath.pi / 6)
            additional_phase_shift = common_impedance.GetAttribute("ag")
            additional_phase_shift_rd = additional_phase_shift * (cmath.pi / 180)
            phase_shift = nominal_phase_shift_rd + additional_phase_shift_rd

            # Get Load-Flow data
            R = common_impedance.GetAttribute("r_pu")
            X = common_impedance.GetAttribute("x_pu")

            new_common_impedance = CommonImpedance(
                data_object=common_impedance,
                name=common_impedance.GetAttribute('loc_name'),
                bus_from=bus_from_name,
                bus_to=bus_to_name,
                R=R,
                X=X,
                U_nominal_HV=busbar_HV_voltage,
                U_nominal_LV=busbar_LV_voltage,
                S_rat=S_rat,
                tap_ratio=tap_ratio,
                phase_shift=phase_shift,
                base_mva=base_mva
            )
            common_impedances.append(new_common_impedance)
        return common_impedances
    
    @staticmethod
    def generate_unique_name(busbar_name: str, substation_name: str) -> str:
        unique_key = f"{substation_name}_{busbar_name}"
        return unique_key
