from ..core.Network import Network
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def build_admittance_matrix(network: Network, as_dataframe: bool = False):
    # Initialize the admittance matrix shape
    Y_bus = np.zeros((len(network.busbars), len(network.busbars)), dtype=complex)

    # Add lines to the admittance matrix
    for line in network.lines:
        if line.busbar_from is None or line.busbar_to is None:
            continue
        idx_from = network.busbar_name_to_index[line.busbar_from]
        idx_to = network.busbar_name_to_index[line.busbar_to]

        Y = line.Y_line
        Y_shunt = line.Y_shunt

        # If line is parallel, calculate new Y & Y_shunt
        if (line.parallel > 1):
            Y        = Y        * line.parallel
            Y_shunt  = Y_shunt  * line.parallel

        Y_bus[idx_from, idx_to] -= Y
        Y_bus[idx_to, idx_from] -= Y
        Y_bus[idx_from, idx_from] += Y + Y_shunt / 2
        Y_bus[idx_to, idx_to] += Y + Y_shunt / 2
    
    # Add generators to the admittance matrix
    for generator in network.synchronous_machines:
        idx = network.busbar_name_to_index[generator.busbar_to]
        try:
            Y_bus[idx, idx] += generator.Y
        except IndexError as e:
            print(f"IndexError at generator {generator.name}: idx={idx}, error={e}")
            raise

    # Add switches to the admittance matrix
    for switch in network.switchs:
        if switch.busbar_from is None or switch.busbar_to is None:
            continue
        idx_from = network.busbar_name_to_index[switch.busbar_from]
        idx_to = network.busbar_name_to_index[switch.busbar_to]
        Y = switch.Y
        Y_bus[idx_from, idx_to] -= Y
        Y_bus[idx_to, idx_from] -= Y
        Y_bus[idx_from, idx_from] += Y
        Y_bus[idx_to, idx_to] += Y
    
    # Add two winding transformers to the admittance matrix
    for transformer in network.two_winding_transformers:
        if transformer.busbar_from is None or transformer.busbar_to is None:
            continue
        idx_from = network.busbar_name_to_index[transformer.busbar_from]   # HV
        idx_to = network.busbar_name_to_index[transformer.busbar_to]       # LV

        Y_elements = transformer.get_admittance_matrix_elements(network.base_mva)
        # Diagonal elements
        Y_bus[idx_from, idx_from] += Y_elements['Y_aa'] * transformer.parallel   
        Y_bus[idx_to, idx_to] += Y_elements['Y_bb'] * transformer.parallel    

        # Off-diagonal elements
        Y_bus[idx_from, idx_to] += Y_elements['Y_ab'] * transformer.parallel
        Y_bus[idx_to, idx_from] += Y_elements['Y_ba'] * transformer.parallel

    # Add three winding transformers to the admittance matrix
    for transformer in network.three_winding_transformers:
        idx_HV = network.busbar_name_to_index[transformer.bus_HV]
        idx_MV = network.busbar_name_to_index[transformer.bus_MV]
        idx_LV = network.busbar_name_to_index[transformer.bus_LV]

        # For simplified version
        Y_delta = transformer.delta_admittance_matrix()
        # Insert into global Y_bus (already sized NxN)
        Y_bus[idx_HV, idx_HV] += Y_delta[0, 0]
        Y_bus[idx_HV, idx_MV] += Y_delta[0, 1]
        Y_bus[idx_HV, idx_LV] += Y_delta[0, 2]
        Y_bus[idx_MV, idx_HV] += Y_delta[1, 0]
        Y_bus[idx_MV, idx_MV] += Y_delta[1, 1]
        Y_bus[idx_MV, idx_LV] += Y_delta[1, 2]
        Y_bus[idx_LV, idx_HV] += Y_delta[2, 0]
        Y_bus[idx_LV, idx_MV] += Y_delta[2, 1]
        Y_bus[idx_LV, idx_LV] += Y_delta[2, 2]
        

    # Add loads to the admittance matrix
    for load in network.loads:
        idx = network.busbar_name_to_index[load.busbar_to]
        Y_bus[idx, idx] += load.Y
    
    # Add Common Impedances to the admittance matrix
    for common_impedance in network.common_impedances:
        idx_from = network.busbar_name_to_index[common_impedance.bus_from]   # HV
        idx_to = network.busbar_name_to_index[common_impedance.bus_to]       # LV
        Y = common_impedance.Y

        # Diagonal elements
        Y_bus[idx_from, idx_from] += Y
        Y_bus[idx_to, idx_to] += Y

        # Off-diagonal elements
        Y_bus[idx_from, idx_to] -= Y
        Y_bus[idx_to, idx_from] -= Y

    # Add External Grids to the admittance matrix
    for external_grid in network.external_grids:
        idx = network.busbar_name_to_index[external_grid.busbar_to]
        Y = external_grid.Y
        Y_bus[idx, idx] += Y

    # Add Voltage Sources to the admittance matrix
    for voltage_source in network.voltage_sources:
        idx = network.busbar_name_to_index[voltage_source.busbar_to]
        Y = voltage_source.Y
        Y_bus[idx, idx] += Y

    # Add Shunts to the admittance matrix
    for shunt in network.shunts:
        idx = network.busbar_name_to_index[shunt.bus_to]
        Y = shunt.Y
        Y_bus[idx, idx] += Y

    if as_dataframe:
        busbar_names = [busbar.name for busbar in network.busbars]
        return pd.DataFrame(Y_bus, index=busbar_names, columns=busbar_names)
    
    return Y_bus

def reduce_matrix(Y_bus: np.ndarray, network: Network):
    '''
    Reduces the admittance matrix Y_bus by eliminating non-generator buses. This is done by:
    1. Extending the admittance matrix to add apparent generator buses
    2. Applying Kron reduction to eliminate non-generator buses

    Args:
        Y_bus (np.ndarray): The admittance matrix to be reduced.
        network (Network): The network object containing busbar and generator information.
    '''
    # Get generator bus names
    gen_bus_names = [gen.busbar_to for gen in network.synchronous_machines if gen.busbar_to is not None] # ['Bus 36G', 'Bus 33G',..]
    is_gen_bus = np.array([1 if bus in gen_bus_names else 0 for bus in network.busbar_name_to_index]) # [0, 0, 1, 0, 1,..] 1 for generator bus

    # Get indices of generator buses and store their name order
    generator_bus_indices = np.where(is_gen_bus == 1)[0] # Get indices of generator buses
    generator_bus_names_order = [network.busbars[i].name for i in generator_bus_indices] # Get names of generator buses

    # ----------- Sort the admittance matrix to have non-generator buses first and generator buses at the end -----------
    # Get all bus names
    all_bus_names =  [bus for bus in network.busbar_name_to_index]

    sorted_idx = sorted(range(len(all_bus_names)), key=lambda i: all_bus_names[i] in gen_bus_names)
    logger.debug(f"Sorted indices to non-generator first and generator second: {sorted_idx}")

    # Reindex Y_red to have non-generator buses first and generator buses second
    Y_bus_sorted = Y_bus[np.ix_(sorted_idx, sorted_idx)]

    # ----------- Build extended version of the admitance matrix -----------
    # Create new apparent generator buses matrix (maintaining the same order as generator_bus_names_order)
    y_gen = np.eye(np.sum(is_gen_bus), dtype=complex)
    for id, gen in enumerate(network.synchronous_machines):
        for i, bus_name in enumerate(generator_bus_names_order):
            if bus_name == gen.busbar_to:
                y_gen[i, i] = gen.Y

    # Connections to the apparent generator buses
    y_gen2 = -y_gen

    # Non-generator buses that are not connected to generator buses
    y_dist = np.zeros((np.sum(is_gen_bus == 1), np.sum(is_gen_bus == 0)))

    # Build the extended admittance matrix
    top_right = np.hstack((y_dist, y_gen2))
    bottom_left = np.vstack((y_dist.T, y_gen2))
    Y_extended = np.block([
        [y_gen,         top_right],
        [bottom_left,   Y_bus_sorted]
    ])
    logger.debug(f"Y_extended shape: {Y_extended.shape}")

    # analyze_extended_matrix_singularities(y_gen, top_right, bottom_left, Y_bus_sorted, network)

    # ----------- Apply Kron reduction to the extended admittance matrix -----------
    num_gen_buses = np.sum(is_gen_bus) # Get number of generator buses
    Y_reduced = KronReduction(Y_extended, num_gen_buses)

    return Y_reduced, generator_bus_names_order

def KronReduction(Y: np.ndarray, p: int):
    '''
    Applies Kron reduction to the given admittance matrix Y.
    Args:
        Y (np.ndarray): The admittance matrix to be reduced.
        p (int): The number of generator buses (the size of the reduced matrix).
    Returns:
        np.ndarray: The reduced admittance matrix.
    '''
    # Extracting sub-blocks
    Y_RR = Y[:p, :p]
    Y_RL = Y[:p, p:]
    Y_LR = Y[p:, :p]
    Y_LL = Y[p:, p:]
    
    # Compute the inverse of Y_LL
    Y_LL_inv = np.linalg.inv(Y_LL)
    
    # Compute Y_reduced using the Schur complement
    Y_reduced = Y_RR - Y_RL @ Y_LL_inv @ Y_LR

    return Y_reduced

def analyze_extended_matrix_singularities(y_gen, top_right, bottom_left, Y_bus_sorted, network: Network):
    """
    Analyze singularities and electrical islands in the extended admittance matrix blocks.
    
    Args:
        y_gen: Generator admittance matrix (top-left block)
        top_right: Top-right block of extended matrix
        bottom_left: Bottom-left block of extended matrix  
        Y_bus_sorted: Bus admittance matrix (bottom-right block)
        network: Network object for bus name mapping
    """
    print("=== EXTENDED MATRIX SINGULARITY ANALYSIS ===\n")
    
    # Get generator and bus information
    gen_bus_names = [gen.busbar_to for gen in network.synchronous_machines if gen.busbar_to is not None]
    all_bus_names = [bus for bus in network.busbar_name_to_index]
    sorted_idx = sorted(range(len(all_bus_names)), key=lambda i: all_bus_names[i] in gen_bus_names)
    non_gen_buses = [all_bus_names[i] for i in sorted_idx if all_bus_names[i] not in gen_bus_names]
    
    # 1. Analyze y_gen matrix (top-left block)
    print("1. GENERATOR MATRIX (y_gen) ANALYSIS:")
    print(f"   Shape: {y_gen.shape}")
    det_y_gen = np.linalg.det(y_gen)
    print(f"   Determinant: {det_y_gen}")
    
    if abs(det_y_gen) < 1e-12:
        print("   ⚠️  SINGULAR! Issues found:")
        for i, val in enumerate(np.diag(y_gen)):
            if abs(val) < 1e-12:
                print(f"     Generator {i}: Y = {val} (ZERO ADMITTANCE)")
    else:
        print("   ✅ Non-singular")
    print()
    
    # 2. Analyze Y_bus_sorted matrix (bottom-right block)
    print("2. BUS ADMITTANCE MATRIX (Y_bus_sorted) ANALYSIS:")
    print(f"   Shape: {Y_bus_sorted.shape}")
    det_Y_bus = np.linalg.det(Y_bus_sorted)
    print(f"   Determinant: {det_Y_bus}")
    
    if abs(det_Y_bus) < 1e-12:
        print("   ⚠️  SINGULAR! Analyzing electrical islands...")
        
        # Find isolated buses (zero diagonal elements)
        isolated_buses = []
        for i, val in enumerate(np.diag(Y_bus_sorted)):
            if abs(val) < 1e-12:
                bus_name = non_gen_buses[i] if i < len(non_gen_buses) else f"Bus_{i}"
                isolated_buses.append((i, bus_name, val))
        
        if isolated_buses:
            print("   🏝️  ISOLATED BUSES (zero diagonal elements):")
            for idx, name, val in isolated_buses:
                print(f"     Bus {idx} ({name}): Y_diag = {val}")
        
        # Find electrical islands using graph connectivity
        islands = find_electrical_islands(Y_bus_sorted, non_gen_buses)
        if len(islands) > 1:
            print(f"   🏝️  ELECTRICAL ISLANDS DETECTED ({len(islands)} islands):")
            for i, island in enumerate(islands):
                island_buses = [non_gen_buses[idx] if idx < len(non_gen_buses) else f"Bus_{idx}" for idx in island]
                print(f"     Island {i+1}: {island_buses}")
        else:
            print("   ✅ No electrical islands detected")
            
        # Check for weakly connected buses
        weak_buses = find_weak_connections(Y_bus_sorted, non_gen_buses)
        if weak_buses:
            print("   ⚠️  WEAKLY CONNECTED BUSES:")
            for idx, name, strength in weak_buses:
                print(f"     Bus {idx} ({name}): connection strength = {strength:.2e}")
    else:
        print("   ✅ Non-singular")
    print()
    
    # 3. Analyze complete extended matrix
    Y_extended = np.block([
        [y_gen,         top_right],
        [bottom_left,   Y_bus_sorted]
    ])
    
    print("3. EXTENDED MATRIX (Y_extended) ANALYSIS:")
    print(f"   Shape: {Y_extended.shape}")
    det_Y_extended = np.linalg.det(Y_extended)
    print(f"   Determinant: {det_Y_extended}")
    
    if abs(det_Y_extended) < 1e-12:
        print("   ⚠️  SINGULAR! This will cause issues in Kron reduction")
        
        # Check rank deficiency
        rank = np.linalg.matrix_rank(Y_extended)
        print(f"   Rank: {rank}/{Y_extended.shape[0]} (deficiency: {Y_extended.shape[0] - rank})")
        
        # Analyze coupling between generator and bus sections
        coupling_strength = np.max(np.abs(top_right)) + np.max(np.abs(bottom_left))
        print(f"   Generator-Bus coupling strength: {coupling_strength:.2e}")
        
    else:
        print("   ✅ Non-singular")
    print()
    
    # 4. Summary and recommendations
    print("4. SUMMARY & RECOMMENDATIONS:")
    if abs(det_y_gen) < 1e-12:
        print("   • Fix generator admittances (ensure Y > 0 for all generators)")
    if abs(det_Y_bus) < 1e-12:
        print("   • Add small shunt admittances to isolated buses")
        print("   • Check network connectivity and remove true islands")
        print("   • Ensure proper grounding for load buses")
    if abs(det_Y_extended) < 1e-12:
        print("   • Extended matrix singularity will cause Kron reduction to fail")
        print("   • Fix underlying singularities in y_gen and Y_bus_sorted first")

def find_electrical_islands(Y_matrix, bus_names):
    """Find electrical islands in the admittance matrix using graph connectivity."""
    n = Y_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    islands = []
    
    def dfs(node, current_island):
        visited[node] = True
        current_island.append(node)
        
        # Check connections to other buses (non-zero off-diagonal elements)
        for neighbor in range(n):
            if not visited[neighbor] and abs(Y_matrix[node, neighbor]) > 1e-12:
                dfs(neighbor, current_island)
    
    for i in range(n):
        if not visited[i]:
            island = []
            dfs(i, island)
            if island:  # Only add non-empty islands
                islands.append(island)
    
    return islands

def find_weak_connections(Y_matrix, bus_names, threshold=1e-6):
    """Find buses with weak electrical connections."""
    weak_buses = []
    
    for i in range(Y_matrix.shape[0]):
        # Calculate connection strength as sum of absolute off-diagonal elements
        row_sum = np.sum(np.abs(Y_matrix[i, :])) - abs(Y_matrix[i, i])
        
        if 0 < row_sum < threshold:
            bus_name = bus_names[i] if i < len(bus_names) else f"Bus_{i}"
            weak_buses.append((i, bus_name, row_sum))
    
    return weak_buses