import powerfactory as pf

def import_pfd_file(app: pf.Application, pfd_file_project: str, pfd_project_name: str) -> pf.DataObject:
    # Get current user folder
    current_user_folder = app.GetCurrentUser()
    if not current_user_folder:
        raise ValueError("Current user folder not found. Please ensure you are logged in to PowerFactory.")
    
    # Create new folder in the current user folder
    new_folder = current_user_folder.CreateObject("IntFolder", "Y Bus Example")

    # Get and set the ComImp command
    ComImp = app.GetFromStudyCase("ComPfdImport")
    ComImp.SetAttribute("g_file", pfd_file_project)
    ComImp.SetAttribute("g_target", new_folder)
    
    project_folder_name = new_folder.GetAttribute("loc_name")
    project_path = project_folder_name + "\\" + pfd_project_name
    ComImp.Execute() # type: ignore

    # Activate the project
    app.ActivateProject(project_path)
    project = app.GetActiveProject()
    project.Activate() # type: ignore

    return project