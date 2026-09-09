import pandas as pd

def main():

    df = pd.read_excel("data/PaHaW/PaHaW_files/corpus_PaHaW.xlsx", engine="openpyxl")
    print(df)
    
if __name__ == "__main__":
    main()