import pandas as pd
import numpy as np
#Data Preprocessing
drivers_df = pd.read_csv("Archive (4)\drivers.csv")
results_df = pd.read_csv('Archive (4)\\results.csv')
driver_standings_df = pd.read_csv("Archive (4)\\driver_standings.csv")

drivers_df = drivers_df.drop(columns=["url", "driverRef"], errors='ignore')

drivers_df["driverId"] = drivers_df["driverId"].astype(int)
results_df["driverId"] = results_df["driverId"].astype(int)

merged_df = pd.merge(results_df, drivers_df, on="driverId", how="left")
merged_df = pd.merge(merged_df, driver_standings_df, on=["raceId", "driverId"], how="left")

print(merged_df.head())
print(merged_df.info())

merged_df.to_csv("results_with_drivers.csv", index=False)
