import powerfactory as pf
from typing import List, Dict, Optional
import logging

from ..utils.Converter import Converter
from ..utils.LoadFlowResults import BusLoadFlowResult

logger = logging.getLogger(__name__)

class Network:
    """
    PowerFactory network reader.
    
    This class reads network data from PowerFactory, classifies elements,
    and provides methods for network analysis.
    """
    # Class constants for busbar types
    BUSBAR_TYPE_BUSBAR = 0
    BUSBAR_TYPE_JUNCTION_NODE = 1
    ACCEPTED_BUSBAR_TYPES = {BUSBAR_TYPE_BUSBAR, BUSBAR_TYPE_JUNCTION_NODE}
    base_mva = 100.0  # Base MVA for the network, can be adjusted as needed

    def __init__(self, app: Optional[pf.Application] = None, base_mva: float = 100.0) -> None:
        """Initialize Network instance."""
        self.base_mva = base_mva
        # Initialize data structures
        self._initialize_data_structures()
        
        # Read network if app is provided
        if app is not None:
            self.read_network_busbars(app)
            self.read_load_flow_results(app)
            self.read_connected_elements(app)
            self.obtain_elements_data()

    def _initialize_data_structures(self) -> None:
        """Initialize all data structures used by the Network class."""
        # Raw PowerFactory objects
        self.PF_busbars: List[pf.DataObject] = []

        # Mapping of busbar names to indices
        self.busbar_name_to_index: Dict[str, int] = {}

        # Load flow results
        self.load_flow_results: Dict[str, BusLoadFlowResult] = {}
        
        # Classified network elements
        self.classified_elements = self._init_classified_elements()

        # Connected elements storage
        self.connected_elements: Dict[str, pf.DataObject] = {}

    def read_network_busbars(self, app: pf.Application) -> None:
        """Read network data from PowerFactory application."""
        try:
            logger.debug("Reading busbars from PowerFactory...")
            pf_busbars = app.GetCalcRelevantObjects("*.ElmTerm", 0, 0, 0)
            
            if not pf_busbars:
                raise ValueError("No busbars found in PowerFactory project")
            
            self.read_busbars(pf_busbars)
            logger.debug(f"Successfully read {len(self.PF_busbars)} busbars")
            
        except Exception as e:
            logger.error(f"Error reading network from PowerFactory: {e}")
            raise ConnectionError(f"Failed to read network from PowerFactory: {e}")

    def read_busbars(self, busbarsPF: List[pf.DataObject]) -> None:
        """Process and filter busbars from PowerFactory."""
        logger.debug(f"Processing {len(busbarsPF)} busbars...")
        
        # Reset data structures
        self.busbar_name_to_index = {}
        self.PF_busbars = []
        self.busbars = []
        filtered_busbars = []

        for busbar in busbarsPF:
            # Remove out of service busbars
            if busbar.GetAttribute("outserv") == 1:
                continue

            # Check busbar type (accept only Busbar - 0 and Junction Node - 1)
            busbar_type = busbar.GetAttribute("iUsage")
            if busbar_type not in self.ACCEPTED_BUSBAR_TYPES:
                continue
            
            # Check if busbar is energized
            if busbar.IsEnergized() != 1:
                continue

            # If goes through all checks, add busbar to the list
            filtered_busbars.append(busbar)
        
        if not filtered_busbars:
            raise ValueError("No valid busbars found after filtering")
        
        # Store filtered PF busbars
        self.PF_busbars = filtered_busbars

        # Store busbar name to index mapping
        for idx, busbar in enumerate(self.PF_busbars):
            busbar_name = busbar.GetAttribute("loc_name")
            if busbar_name in self.busbar_name_to_index:
                logger.warning(f"Duplicate busbar name found: {busbar_name}")
            
            self.busbar_name_to_index[busbar_name] = idx

        # Store busbars as objects (commented out until Converter is available)
        self.busbars = Converter.convert_busbars(self.PF_busbars)
        
        logger.info(f"Processed {len(self.PF_busbars)} valid busbars")

    def read_connected_elements(self, app: pf.Application) -> None:
        """Read all connected elements to the busbars and classify them."""
        try:
            self.classified_elements = self._init_classified_elements()
            logger.info("Reading network elements from PowerFactory...")
            
            # Define elements to read directly
            elements_to_read = {
                'ElmLne': 'Lines',
                'ElmTr2': 'Transformers 2-Winding', 
                'ElmLod': 'Loads',
                'ElmSym': 'Synchronous Machines',
                # 'ElmCoup': 'Switches', # Will be read separately since too many elements
                'ElmTr3': 'Transformers 3-Winding',
                'ElmZpu': 'Common Impedance',
                'ElmVac': 'AC Voltage Source',
                'ElmShnt': 'Switchable Shunt',
                'ElmXnet': 'External Grid',
                'ElmIac': 'AC Current Source',
                'ElmSind': 'Series Reactors',
                'ElmScap': 'Series Capacitors',
                'ElmBranch': 'Branch Elements',
                'ElmSssc': 'Static Synchronous Series Compensator',
                'ElmSubmodcon': 'Submodule Connector',
                'ElmVacbin': 'Binary AC Voltage Source',
                'ElmVscmono': 'PWM Converter',
                'ElmGenstat': 'Static Generator',
                'ElmSvs': 'Static Var System'
            }
            
            total_elements = 0
            
            for element_class, description in elements_to_read.items():
                try:
                    # Get all elements of this class
                    pf_elements = app.GetCalcRelevantObjects(f"*.{element_class}", 1, 1, 1)
                    
                    if pf_elements:
                        filtered_elements = []
                        
                        for element in pf_elements:
                            # Check if element is in service
                            if element.GetAttribute("outserv") == 1:
                                continue
                            
                            # Check if element is energized
                            if element.IsEnergized() != 1:
                                continue

                            # Get unique name for connecting circuits breakers between double busbars
                            if element_class == "ElmCoup":
                                if (element.GetAttribute('aUsage') != "cbk"):
                                    continue
                                # substation = element.GetAttribute('cpSubstat')
                                # substation_name = substation.GetAttribute('loc_name') if substation is not None else "None"
                                # element_id = Converter.generate_unique_name(element_id, substation_name)
                            
                            filtered_elements.append(element)
                        
                        self.classified_elements[element_class] = filtered_elements
                        total_elements += len(filtered_elements)
                        logger.debug(f"Read {len(filtered_elements)} {description}")
                    
                except Exception as e:
                    logger.warning(f"Error reading {element_class}: {e}")
                    self.classified_elements[element_class] = []

            # Read switches from busbars separately
            self.read_switches_from_busbars()
            
            logger.info(f"Successfully read {total_elements} classified elements from PF")
            
            # Log classified element counts if logger level is INFO or lower
            if logger.isEnabledFor(logging.INFO):
                logger.info("Classified element counts:")
                for element_type, elements_list in self.classified_elements.items():
                    if elements_list:  # Only log if there are elements of this type
                        logger.info(f"  {element_type}: {len(elements_list)}")
            
            if total_elements == 0:
                logger.warning("No classified elements found in the network")
            
        except Exception as e:
            logger.error(f"Error reading classified elements: {e}")
            raise ValueError(f"Failed to read classified elements: {e}")
        
    def read_switches_from_busbars(self) -> None:
        checked_switches = set()  # To avoid duplicates

        for busbar in self.PF_busbars:
            # Get all connected elements to the busbar
            connected_elements: List[pf.DataObject] = busbar.GetConnectedElements(1, 1, 1)

            for element in connected_elements:
                if element.GetClassName() == "ElmCoup":
                    element_id = element.GetAttribute('loc_name')
                    substation: pf.DataObject = element.GetAttribute('cpSubstat')
                    substation_name = substation.GetAttribute('loc_name') if substation is not None else "None"
                    element_id = Converter.generate_unique_name(element_id, substation_name)

                    # Check if the switch has already been processed
                    if element_id in checked_switches:
                        # logger.debug(f"Switch {element_id} already processed, skipping...")
                        continue

                    # Check if element is in service
                    if element.GetAttribute("outserv") == 1:
                        continue

                    # Check if element is energized
                    if element.IsEnergized() != 1:
                        continue
                    
                    # Add the switch to the checked set
                    checked_switches.add(element_id)

                    # Add the switch to the classified elements
                    self.classified_elements['ElmCoup'].append(element)

    def obtain_elements_data(self):
        '''
        Converts classified elements from PF Objects to internal objects and stores them in self.<element_type>
        '''
        logger.debug("Converting classified elements to internal objects...")

        # Convert supported classified elements to internal objects
        self.lines = Converter.convert_lines(self.classified_elements['ElmLne'], self.busbar_name_to_index, base_mva=self.base_mva)
        self.loads = Converter.convert_loads(self.classified_elements['ElmLod'], self.load_flow_results, base_mva=self.base_mva)
        self.switchs = Converter.convert_switches(self.classified_elements['ElmCoup'], base_mva=self.base_mva)
        self.two_winding_transformers = Converter.convert_two_winding_transformers(self.classified_elements['ElmTr2'], base_mva=self.base_mva)
        self.three_winding_transformers = Converter.convert_three_winding_transformers(self.classified_elements['ElmTr3'], base_mva=self.base_mva)
        self.synchronous_machines = Converter.convert_synchronous_machines(self.classified_elements['ElmSym'], base_mva=self.base_mva)
        self.common_impedances = Converter.convert_common_impedances(self.classified_elements['ElmZpu'], base_mva=self.base_mva)
        self.external_grids = Converter.convert_external_grids(self.classified_elements['ElmXnet'], base_mva=self.base_mva)
        self.voltage_sources = Converter.convert_voltage_source_AC(self.classified_elements['ElmVac'], base_mva=self.base_mva)
        self.shunts = Converter.convert_shunt_elements(self.classified_elements['ElmShnt'], base_mva=self.base_mva)

    def _init_classified_elements(self) -> Dict[str, List[pf.DataObject]]:
        # Only supported elements are initialized here
        return {
            'ElmLne': [],       # Lines
            'ElmTr2': [],       # Transformers 2-Winding
            'ElmTr3': [],       # Transformers 3-Winding
            'ElmLod': [],       # Loads
            'ElmSym': [],       # Synchronous Machines
            'ElmCoup': [],      # Switches
            'ElmZpu': [],       # Common impedance
            'ElmVac': [],       # AC Voltage Source
            'ElmShnt': [],      # Switchable Shunt # Partial supoport
            'ElmXnet': [],      # External grid
        }

    def read_load_flow_results(self, app: pf.Application) -> None:
        logger.info("Running load flow analysis to obtain busbar results...")
        # Create a load flow object and execute as we need the load flow results
        ldf: pf.DataObject = app.GetFromStudyCase("ComLdf")
        ldf.Execute() # type: ignore 
        bus_load_flow_results: Dict[str, BusLoadFlowResult] = {}
        for busbar in self.PF_busbars:
            name = busbar.GetAttribute("loc_name")

            try:
                voltage = busbar.GetAttribute("m:u")
                angle = busbar.GetAttribute("m:phiu")
            except Exception as e:
                print(f"Error while reading busbar {name}: {e}")
                continue
            
            bus_load_flow_results[name] = BusLoadFlowResult(
                name=name,
                voltage=voltage,
                angle=angle
            )

        # Store the load flow results
        if not bus_load_flow_results:
            raise ValueError("No load flow results found for busbars")
        logger.debug(f"Successfully obtained load flow results for {len(bus_load_flow_results)} busbars")
        self.load_flow_results = bus_load_flow_results

    @property
    def busbar_count(self) -> int:
        """Get the number of valid busbars in the network."""
        return len(self.PF_busbars)
    
    def get_busbar_index(self, busbar_name: str) -> Optional[int]:
        return self.busbar_name_to_index.get(busbar_name)
    
    def get_generator_name_from_busbar(self, busbar_name: str) -> str:
        for gen in self.synchronous_machines:
            if gen.busbar_to == busbar_name:
                return gen.name
        raise ValueError(f"No generator found for busbar '{busbar_name}'")
    
    def get_load_flow_results(self) -> Dict[str, BusLoadFlowResult]:
        """Get the load flow results for all busbars."""
        return self.load_flow_results