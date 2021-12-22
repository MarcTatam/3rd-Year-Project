import pandas as pd

def open_cdr(date:str)->pd.DataFrame:
    df = pd.read_table("./dataverse_files/sms-call-internet-mi-"+ date +".txt",header=None)
    return df

def open_all_cdr()->[pd.DataFrame]:
    dfs =[]
    for i in range(1,31):
        dfs.append(open_cdr("2013-11-" + str(i).zfill(2)))
    for i in range(1,32):
        dfs.append(open_cdr("2013-12-" + str(i).zfill(2)))
    return dfs

def merge_countries(date:str)->pd.DataFrame:
    df = open_cdr(date)
    df = df.drop(columns=[2])
    df = df.groupby(by=[0,1])
    df = df.sum()
    return df.reset_index()

def merge_all()->pd.DataFrame:
    df = merge_countries("2013-11-01")
    for i in range(2,31):
        temp_df = merge_countries("2013-11-" + str(i).zfill(2))
        df = pd.concat([df,temp_df])
    for i in range(1,32):
        temp_df = merge_countries("2013-12-" + str(i).zfill(2))
        df = pd.concat([df,temp_df])
    return df

if __name__ == "__main__":
    df = merge_countries()
