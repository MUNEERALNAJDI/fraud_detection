from src.data_prep import load_config, load_all_data, clean_cdr

def main():
    config = load_config("configs/baseline.yaml")
    data = load_all_data(config, nrows=config["training"]["sample_size"])
    
    # Example: clean and print shape
    for name, df in data.items():
        df_clean = clean_cdr(df)
        print(f"{name}: {df_clean.shape}")

if __name__ == "__main__":
    main()
