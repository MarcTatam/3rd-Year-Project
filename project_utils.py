def convert_to_project(date: str)->int:
    """Converts from DD/MM/YYYY string to a int representing the day relative to project time
    
    Args
    date - date in DD/MM/YYYY format
    
    Returns
    int representing the day in project time space"""
    if date[3:5] == "12":
        return 29 + int(date[0:2])
    else:
        return int(date[0:2])-1

def convert_to_project_hourly(date:str):
    """Converts from DD/MM/YYYY string to a int representing the day relative to project time
    
    Args
    date - date in DD/MM/YYYY format
    
    Returns
    an int representing the start and end of the day in project time space"""
    if date[3:5] == "12":
        return (29 + int(date[0:2]))*24, (29 + int(date[0:2]))*24+23
    else:
        return (int(date[0:2])-1)*24,(int(date[0:2])-1)*24+23
