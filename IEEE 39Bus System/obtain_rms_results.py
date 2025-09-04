import powerfactory as pf
from typing import List
from powerfactory import DataObject
import time
import sys
import os

def obtain_rms_results(app: pf.Application, folder_name: str):
    # Get the ElmRes object
    elmRes = app.GetCalcRelevantObjects("*All calculations.ElmRes")[0]
    print("Ime Rezultatov: ", elmRes.GetAttribute("loc_name"))
    
    all_generators = app.GetCalcRelevantObjects("*.ElmSym", 1, 1, 1)
    print("All generators: ", len(all_generators))
    generators: List[DataObject] = []
    for gen in all_generators:
        if gen.GetAttribute("outserv") == 1:
            continue
        if gen.IsEnergized() != 1:
            continue
        generators.append(gen)
    print("Generators: ", len(generators))
    print("Generators: ", [gen.GetAttribute("loc_name") for gen in generators])

    for gen in generators:
        obj = elmRes.CreateObject("IntMon")
        obj.SetAttribute("loc_name", gen.GetAttribute("loc_name"))
        obj.SetAttribute("obj_id", gen)
        selected_variables = ["s:P1"]
        obj.SetAttribute("vars", selected_variables)

    # DISABLE ALL CURRENT SIMULATION EVENTS (set as out of service)
    active = app.GetActiveStudyCase()
    eventFolder: DataObject = next((con for con in active.GetContents() if con.loc_name == "Simulation Events/Fault"))
    simulation_events: List[DataObject] = eventFolder.GetContents()
    for event in simulation_events:
        event.SetAttribute("outserv", 1)

    iteration = 0
    created_events = []
    previous_event = None
    files_names = []

    for gen in generators:
        iteration += 1

        gen_name = gen.GetAttribute("loc_name")
        print(f"Processing generator outage for generator nr. {iteration}: {gen_name}")

        # ====== 1. We define the SwitchEvent
        new_event = eventFolder.CreateObject("EvtSwitch")
        new_event.SetAttribute("loc_name", gen_name+"_izpad")
        new_event.SetAttribute("p_target", gen)
        new_event.SetAttribute("time", 2)
        created_events.append(new_event)

        # ====== 2. We run the simulation
        # Calculate initial conditions
        oInit = app.GetFromStudyCase('ComInc')  # Get initial condition calculation object
        oInit.Execute() # type: ignore

        # Run RMS-simulation
        oRms = app.GetFromStudyCase('ComSim')   # Get RMS-simulation object
        oRms.Execute() # type: ignore

        # ====== 3. We delete the current event if it exists
        if previous_event is not None:
            deleted = previous_event.Delete()
            if deleted == 0:
                pass
            else:
                print("Failed to delete event: ", previous_event.GetAttribute("loc_name"))
        new_event.SetAttribute("outserv", 1)

        # ====== 4. We get the results
        comRes = app.GetFromStudyCase("ComRes")
        comRes.pResult = elmRes         # Set ElmRes object to export from # type: ignore
        comRes.iopt_exp = 6             # Set export to csv - 6 # type: ignore
        comRes.iopt_sep = 0             # Set use the system seperator # type: ignore
        comRes.iopt_honly = 0           # To export data and not only the head er # type:ignore
        comRes.iopt_csel = 1            # Set export to only selected variables # type: ignore
        comRes.numberPrecisionFixed = 8 # Set the number precision to 6 decimals # type: ignore
        comRes.col_Sep = ";"            # Set the column separator to ; # type: ignore
        comRes.dec_Sep = ","            # Set the decimal separator to , # type: ignore

        # Set File path and name 
        file_name = "results_izpad_" + gen_name + ".csv"
        files_names.append(file_name)
        results_folder = f"C:\\Projects\\Magistrska Official Repo\\IEEE 39Bus System\\{folder_name}"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        file_path = os.path.join(results_folder, file_name)
        comRes.f_name = file_path # type: ignore

        resultObject = [None] # type: ignore
        elements = [elmRes] # type: ignore
        variable = ["b:tnow"] # type: ignore

        for gen in generators:
            resultObject.append(None)
            elements.append(gen)
            variable.append("s:P1")
        
        # Set the selected variables
        comRes.resultobj = resultObject # Export selected # type: ignore
        comRes.element = elements # type: ignore
        comRes.variable = variable # type: ignore

        # Export the results
        comRes.Execute() # type: ignore 

        # Await for the file to be accessible
        while not os.path.exists(file_path):
            time.sleep(0.1)

        # ====== 5. We disable the SwitchEvent (set to out of service here as cannot delete, will be deleted in the next iteration)
        new_event.SetAttribute("outserv", 1)
        previous_event = new_event

        app.ClearOutputWindow()