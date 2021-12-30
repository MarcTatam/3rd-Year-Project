import pandas as pd

def open_cdr(date:str)->pd.DataFrame:
    """Loads the CDR data for a given date
    
    Args
    date - the date to load in string form
    
    Returns
    Dataframe with the activity data"""
    df = pd.read_table("./dataverse_files/sms-call-internet-mi-"+ date +".txt",header=None)
    return df

def open_all_cdr()->[pd.DataFrame]:
    """Loads the CDR data for all dates
    Deprecated
    
    Returns
    List of data frames for each day"""
    dfs =[]
    for i in range(1,31):
        dfs.append(open_cdr("2013-11-" + str(i).zfill(2)))
    for i in range(1,32):
        dfs.append(open_cdr("2013-12-" + str(i).zfill(2)))
    return dfs

def merge_countries(date:str)->pd.DataFrame:
    """Opens the CDR for a given date and merges calls from different country codes
    
    Args
    date - The date to load in string form
    
    Returns
    Dataframe for the given date"""
    df = open_cdr(date)
    df = df.drop(columns=[2])
    df = df.groupby(by=[0,1])
    df = df.sum()
    return df.reset_index()

def merge_all()->pd.DataFrame:
    """Opens all CDRs and merges them into a single dataframe
    
    Returns
    Dataframe containing all CDRs"""
    df = merge_countries("2013-11-01")
    for i in range(2,31):
        print(i)
        temp_df = merge_countries("2013-11-" + str(i).zfill(2))
        df = pd.concat([df,temp_df])
    for i in range(1,32):
        print(i)
        temp_df = merge_countries("2013-12-" + str(i).zfill(2))
        df = pd.concat([df,temp_df])
    return df

if __name__ == "__main__":
    df = merge_all()
    print(df)
    df[2] = df[3]+df[4]
    print(df)
    print(df[2].min())
