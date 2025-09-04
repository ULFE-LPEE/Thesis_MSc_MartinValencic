import powerfactory as pf

def init_project(app: pf.Application, project_path: str):
    # Get current user folder
    current_user_folder = app.GetCurrentUser()
    if not current_user_folder:
        raise ValueError("Current user folder not found. Please ensure you are logged in to PowerFactory.")

    # Activate the existing project
    try:
        app.ActivateProject(project_path)
        project = app.GetActiveProject()
        if project:
            project.Activate() # type: ignore
            print(f"Successfully activated project: {project.GetAttribute('loc_name')}")
        else:
            raise ValueError("Failed to get active project after activation")
    except Exception as e:
        print(f"Error activating project: {e}")