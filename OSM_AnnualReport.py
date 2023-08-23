"""
OSM Annual Reporting
    Purpose: automate production of annual report figures and tables

Requires user to specify reporting year (could be set instead as an input option when running the script)

Required local files:
    Hydat.sqlite3 - download most up to date version here: https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/
    Station_Meta.csv - station ID, station name, start of record, RAMP record, which years of RAMP to include in mean
                        (ie. >=365 days of data available), GDA climate station to use for figures (could change in
                        future years, depending on which is closest/has available data)
    RAMP_Q.csv - RAMP daily flow data to add to HYDAT daily record (for all applicable stations, will not change from
                    year to year)
    RAMP_H.csv - RAMP daily level data to add to HYDAT daily record (for all applicable stations, will not change from
                    year to year)
    DailyPrecipYYYY.csv - daily precip record (e.g. YYYY=2018) for ACIS, RAMP and WBEA climate stations (will need to
                            create each year, at least until there is a suitable ECCC climate station nearby each
                            hydrometric station with data available)

@author: Regan Willenborg, regan.willenborg@ec.gc.ca, RW
         James Leach, james.leach@ec.gc.ca, JL

"""
###################################################################
###                 Required Modules/Libraries                  ###
###################################################################
import datetime as dt
import os
import sqlite3
from io import StringIO
from timeit import default_timer as timer

import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sigfig import round

try:
    from osgeo import gdal
except ModuleNotFoundError:
    import gdal



###################################################################
###         OSM Stations, File Locations, & other setup         ###
###################################################################
# Use RDPA?
use_RDPA = True
ignore_current_year_for_stats = False

# Key Stations for Main Body Report
stn = ['07DD001', '07DA001', '07CD005', '07BE001']

# All Oil Sands Hydrometric Stations to be included in Appendices
flow_stn = ['07BE001', '07DD001', '07DA001', '07DA018', '07DA040', '07DA033', '07CE013', '07CE002', '07CE007',
            '07CD005', '07CD001', '07DB002', '07DB003', '07DA038', '07DA039', '07DA032', '07DA041', '07DC001',
            '07DC003', '07CE008', '07CD004', '07CD008', '07CD009', '07CB002', '07DA027', '07CE005', '07DA026',
            '07DA030', '07DB006', '07DB001', '07DC004', '07DA035', '07DA029', '07DA028', '07DA008', '07DA034',
            '07CE003', '07DA007', '07DA042', '07DA044', '07DA006', '07CE010', '07DA037', '07DA045']

level_stn = ['07CE001', '07DA024', '07DA023', '07DA025']

# Make sure files are saved in current working directory 
files_dir = os.getcwd()

# Locally Saved Data
# Set up connection to HYDAT database saved to local folder
database = r"C:\Program Files (x86)\ECCC\ECDataExplorer\Database\Hydat.sqlite3"
conn = sqlite3.connect(database)

# Station Meta Data (record, RAMP record, associated climate stations)
df_meta = pd.read_csv(os.path.join(files_dir, 'Station_Meta.csv'), low_memory=False)
dfCSL = pd.read_csv('Climate_station_locations.csv')

# RAMP data
df_RAMP_Q = pd.read_csv(os.path.join(files_dir, 'RAMP_Q.csv'), low_memory=False)
df_RAMP_H = pd.read_csv(os.path.join(files_dir, 'RAMP_H.csv'), low_memory=False)

###################################################################
###                 Data Collection and Analysis                ###
###################################################################
def get_stn_name(conn, stn):
    """Function to get station name from HYDAT database
    (returns string)"""
    df = pd.read_sql_query(f"SELECT STATION_NAME FROM STATIONS WHERE STATION_NUMBER = '{stn}'", conn)
    stn_name = df.iloc[0]['STATION_NAME']
    return stn_name


def get_record_period_Q(conn, stn):
    """Function to get station period of record (flow measurements) from combined HYDAT database and RAMP data
    (returns dataframe)"""
    df = pd.read_sql_query(
        f"SELECT STATION_NUMBER, YEAR_FROM, YEAR_TO FROM STN_DATA_RANGE WHERE STATION_NUMBER = '{stn}' AND DATA_TYPE = 'Q'"
        , conn)
    # import record/station information (RAMP, HYDAT, Drainage Areas)
    # df_meta = pd.read_csv(os.path.join(files_dir, 'Station_Meta.csv'), low_memory=False)

    df["YEAR_FROM"] = df_meta[df_meta['ID'] == stn]["Full Record Start"].iloc[0]
    if ignore_current_year_for_stats:
        df["YEAR_TO"] = previous_yr
    else:
        df["YEAR_TO"] = current_yr

    return df


def get_record_period_H(conn, stn):
    """Function to get station period of record (level measurements) from combined HYDAT database and RAMP data
    (returns dataframe)"""
    df = pd.read_sql_query(
        f"SELECT STATION_NUMBER, YEAR_FROM, YEAR_TO FROM STN_DATA_RANGE WHERE STATION_NUMBER = '{stn}' AND DATA_TYPE = 'H'"
        , conn)
    # import record/station information (RAMP, HYDAT, Drainage Areas)
    # df_meta = pd.read_csv(os.path.join(files_dir, 'Station_Meta.csv'), low_memory=False)

    df["YEAR_FROM"] = df_meta[df_meta['ID'] == stn]["Full Record Start"].iloc[0]
    if ignore_current_year_for_stats:
        df["YEAR_TO"] = previous_yr
    else:
        df["YEAR_TO"] = current_yr

    return df


# Flow Data
def get_daily_flows(conn, stn):
    """Function to pull daily flow data from HYDAT and RAMP
    (returns dataframe)"""
    # Get daily flows and flags from hydat
    df_raw = pd.read_sql_query(f"SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER = '{stn}'", conn)

    # Break tables
    basecolumn = ['STATION_NUMBER', 'YEAR', 'MONTH']
    column_flow_day = basecolumn + [f'FLOW{i}' for i in range(1, 32)]
    column_symbol_day = basecolumn + [f'FLOW_SYMBOL{i}' for i in range(1, 32)]

    df_flow = df_raw[column_flow_day]
    df_flag = df_raw[column_symbol_day]

    # Unpivot Daily Flows
    df_flow_melt = pd.melt(df_flow,
                           id_vars=['STATION_NUMBER', 'YEAR', 'MONTH'],
                           var_name='DAY',
                           value_name="FLOW").sort_values(by=['YEAR', 'MONTH'])

    # Unpivot Daily Symbols
    df_flag_melt = pd.melt(df_flag,
                           id_vars=['STATION_NUMBER', 'YEAR', 'MONTH'],
                           var_name='FlowSymbol',
                           value_name="FLAG").sort_values(by=['YEAR', 'MONTH'])
    df_flag_melt['DAY'] = df_flag_melt['FlowSymbol'].apply(lambda s: 'FLOW' + s.split('FLOW_SYMBOL')[1])

    # Join Tables
    df_HYDAT = df_flow_melt.merge(df_flag_melt,
                                  left_on=['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY'],
                                  right_on=['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY'],
                                  how='inner', suffixes=('', '_'))

    # Drop/rename columns
    df_HYDAT.drop(['FlowSymbol'], axis=1, inplace=True)
    df_HYDAT['DAY'] = df_HYDAT['DAY'].apply(lambda d: int(d.split('FLOW')[1]))

    # df = df.dropna(subset = ['FLOW'])
    df_HYDAT.reset_index(level=None, drop=True, inplace=True, col_level=0)

    # Add Date
    df_HYDAT['DATE'] = pd.to_datetime(df_HYDAT[['YEAR', 'MONTH', 'DAY']], errors='coerce')

    # Add Ramp data, if it exists
    df_RAMP = df_RAMP_Q.filter(['Station', 'Date', 'Discharge', 'Discharge Flag'])
    df_RAMP = df_RAMP.rename(
        columns={'Station': 'STATION_NUMBER', 'Date': 'DATE', 'Discharge': 'FLOW', 'Discharge Flag': 'FLAG'})
    df_RAMP['DATE'] = pd.to_datetime(df_RAMP['DATE']).dt.normalize()
    df_RAMP['YEAR'] = df_RAMP['DATE'].dt.year
    df_RAMP['MONTH'] = df_RAMP['DATE'].dt.month
    df_RAMP['DAY'] = df_RAMP['DATE'].dt.day
    cols = ['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY', 'FLOW', 'FLAG', 'DATE']
    df_RAMP = df_RAMP[cols]
    df_RAMP = df_RAMP[df_RAMP['STATION_NUMBER'] == stn]

    # Prioritize HYDAT if there is overlap in data
    df_RAMP = df_RAMP[~df_RAMP['DATE'].isin(df_HYDAT['DATE'])]

    # Merge RAMP with HYDAT
    frames = [df_RAMP, df_HYDAT]
    df = pd.concat(frames, ignore_index=True)

    # Convert types
    df.YEAR = df.YEAR.astype(int)
    df.MONTH = df.MONTH.astype(int)
    df.DAY = df.DAY.astype(int)

    # Remove data after current year
    df = df[df.YEAR <= current_yr]

    return df


def get_mean_flows(conn, stn):
    """Function to calculate annual mean flows
    (returns dataframe)"""
    daily = get_daily_flows(conn, stn)
    annual_mean = pd.DataFrame({"MEAN": daily.groupby('YEAR')['FLOW'].mean()})
    annual_mean["Count"] = daily.groupby('YEAR')['FLOW'].count()
    annual_mean.reset_index(level=None, drop=False, inplace=True, col_level=0, )

    # only return years where there are no missing days of data (consistent with WSC reporting)
    return annual_mean[annual_mean.Count >= 365]


def get_quantiles_flows(df):
    """Function to calculate percentiles for each day in the year, based on entire period of record
    (returns dataframe)"""
    df_record = df.copy()

    if ignore_current_year_for_stats:
        df_record = df_record[df_record['YEAR'] != current_yr]

    df_count = df_record.groupby(['MONTH', 'DAY'])['FLOW'].count()
    df_count = df_count.reset_index(level=None, drop=False)

    df_min = df_record.groupby(['MONTH', 'DAY'])['FLOW'].min()
    df_min = df_min.reset_index(level=None, drop=False)

    df_25 = df_record.groupby(['MONTH', 'DAY'])['FLOW'].quantile(0.25)
    df_25 = df_25.reset_index(level=None, drop=False)

    df_50 = df_record.groupby(['MONTH', 'DAY'])['FLOW'].quantile(0.50)
    df_50 = df_50.reset_index(level=None, drop=False)

    df_75 = df_record.groupby(['MONTH', 'DAY'])['FLOW'].quantile(0.75)
    df_75 = df_75.reset_index(level=None, drop=False)

    df_max = df_record.groupby(['MONTH', 'DAY'])['FLOW'].max()
    df_max = df_max.reset_index(level=None, drop=False)

    df = pd.DataFrame([df_min.MONTH,
                       df_min.DAY,
                       df_count.FLOW,
                       df_min.FLOW,
                       df_25.FLOW,
                       df_50.FLOW,
                       df_75.FLOW,
                       df_max.FLOW]).transpose()
    df.columns = ['MONTH', 'DAY', 'n', 'MIN', 'p_25', 'p_50', 'p_75', 'MAX']
    df['YEAR'] = np.full((len(df.MAX), 1), current_yr)

    """if (current_yr % 4) == 0:
        if (current_yr % 100) == 0:
            if (current_yr % 400) == 0:
                pass  # leap year
            else:
                df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)  # remove February 29, not a leap year
        else:
            pass  # leap year
    else:
        df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)  # remove February 29, not a leap year"""
    # always drop Feb. 29 -> little data on that day makes bounds look strange
    df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)

    df.reset_index(level=None, drop=True, inplace=True, col_level=0)
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')

    # only return days where there are 2 or more years of measurements on that day
    return df[df.n >= 2]


# Level Data
def get_daily_levels(conn, stn):
    """Function to pull daily level data from HYDAT and RAMP
    (returns dataframe)"""
    # Get daily flows and flags from hydat
    df_raw = pd.read_sql_query(f"SELECT * FROM DLY_LEVELS WHERE STATION_NUMBER = '{stn}'", conn)

    # Break tables
    basecolumn = ['STATION_NUMBER', 'YEAR', 'MONTH']
    column_level_day = basecolumn + [f'LEVEL{i}' for i in range(1, 32)]
    column_symbol_day = basecolumn + [f'LEVEL_SYMBOL{i}' for i in range(1, 32)]

    df_level = df_raw[column_level_day]
    df_flag = df_raw[column_symbol_day]

    # Unpivot Daily Flows
    df_level_melt = pd.melt(df_level,
                            id_vars=['STATION_NUMBER', 'YEAR', 'MONTH'],
                            var_name='DAY',
                            value_name="LEVEL").sort_values(by=['YEAR', 'MONTH'])

    # Unpivot Daily Symbols
    df_flag_melt = pd.melt(df_flag,
                           id_vars=['STATION_NUMBER', 'YEAR', 'MONTH'],
                           var_name='LevelSymbol',
                           value_name="FLAG").sort_values(by=['YEAR', 'MONTH'])
    df_flag_melt['DAY'] = df_flag_melt['LevelSymbol'].apply(lambda s: 'LEVEL' + s.split('LEVEL_SYMBOL')[1])

    # Join Tables
    df_HYDAT = df_level_melt.merge(df_flag_melt,
                                   left_on=['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY'],
                                   right_on=['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY'],
                                   how='inner', suffixes=('', '_'))

    # Drop/rename columns
    df_HYDAT.drop(['LevelSymbol'], axis=1, inplace=True)
    df_HYDAT['DAY'] = df_HYDAT['DAY'].apply(lambda d: int(d.split('LEVEL')[1]))

    # df = df.dropna(subset = ['LEVEL'])
    df_HYDAT.reset_index(level=None, drop=True, inplace=True, col_level=0)

    # Add Date
    df_HYDAT['DATE'] = pd.to_datetime(df_HYDAT[['YEAR', 'MONTH', 'DAY']], errors='coerce')

    # Add Ramp data, if it exists
    df_RAMP = df_RAMP_H.filter(['Station', 'Date', 'Level', 'Level Flag'])
    df_RAMP = df_RAMP.rename(
        columns={'Station': 'STATION_NUMBER', 'Date': 'DATE', 'Level': 'LEVEL', 'Level Flag': 'FLAG'})
    df_RAMP['DATE'] = pd.to_datetime(df_RAMP['DATE']).dt.normalize()
    df_RAMP['YEAR'] = df_RAMP['DATE'].dt.year
    df_RAMP['MONTH'] = df_RAMP['DATE'].dt.month
    df_RAMP['DAY'] = df_RAMP['DATE'].dt.day
    cols = ['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY', 'LEVEL', 'FLAG', 'DATE']
    df_RAMP = df_RAMP[cols]
    df_RAMP = df_RAMP[df_RAMP['STATION_NUMBER'] == stn]

    # Prioritize HYDAT if there is overlap in data
    df_RAMP = df_RAMP[~df_RAMP['DATE'].isin(df_HYDAT['DATE'])]

    # Merge RAMP with HYDAT
    frames = [df_RAMP, df_HYDAT]
    df = pd.concat(frames, ignore_index=True)

    # Account for datum shift
    if stn == '07DA024':
        df = df[(df['DATE'] > dt.datetime(2017, 10, 21))]

    # Convert types
    df.YEAR = df.YEAR.astype(int)
    df.MONTH = df.MONTH.astype(int)
    df.DAY = df.DAY.astype(int)

    # Remove data after current year
    df = df[df.YEAR <= current_yr]
    return df


def get_mean_levels(conn, stn):
    """Function to calculate annual mean level for each year in record
    (returns dataframe)"""
    daily = get_daily_levels(conn, stn)
    annual_mean = pd.DataFrame({"MEAN": daily.groupby('YEAR')['LEVEL'].mean()})
    annual_mean["Count"] = daily.groupby('YEAR')['LEVEL'].count()
    annual_mean.reset_index(level=None, drop=False, inplace=True, col_level=0, )

    # only return years where there are no missing days of data
    return annual_mean[annual_mean.Count >= 365]


def get_quantiles_levels(df):
    """Function to calculate percentiles for each day in the year, based on entire period of record
    (returns dataframe)"""
    df_record = df.copy()

    if ignore_current_year_for_stats:
        df_record = df_record[df_record['YEAR'] != current_yr]

    df_count = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].count()
    df_count = df_count.reset_index(level=None, drop=False)

    df_min = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].min()
    df_min = df_min.reset_index(level=None, drop=False)

    df_25 = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].quantile(0.25)
    df_25 = df_25.reset_index(level=None, drop=False)

    df_50 = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].quantile(0.50)
    df_50 = df_50.reset_index(level=None, drop=False)

    df_75 = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].quantile(0.75)
    df_75 = df_75.reset_index(level=None, drop=False)

    df_max = df_record.groupby(['MONTH', 'DAY'])['LEVEL'].max()
    df_max = df_max.reset_index(level=None, drop=False)

    df = pd.DataFrame([df_min.MONTH,
                       df_min.DAY,
                       df_count.LEVEL,
                       df_min.LEVEL,
                       df_25.LEVEL,
                       df_50.LEVEL,
                       df_75.LEVEL,
                       df_max.LEVEL]).transpose()
    df.columns = ['MONTH', 'DAY', 'n', 'MIN', 'p_25', 'p_50', 'p_75', 'MAX']
    df['YEAR'] = np.full((len(df.MAX), 1), current_yr)

    '''if (current_yr % 4) == 0:
        if (current_yr % 100) == 0:
            if (current_yr % 400) == 0:
                pass  # leap year
            else:
                df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)  # remove February 29, not a leap year
        else:
            pass  # leap year
    else:
        df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)  # remove February 29, not a leap year'''
    # always drop Feb. 29 -> little data on that day makes bounds look strange
    df = df.drop(df[(df["MONTH"] == 2) & (df["DAY"] == 29)].index)

    df.reset_index(level=None, drop=True, inplace=True, col_level=0)
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')

    # only return days where there are 2 or more years of measurements on that day
    return df[df.n >= 2]


# Ice affected Period
def get_ice_affected_period(df_cy):
    """Function to define open water vs ice affected for current year"""
    df = df_cy.copy()
    df['VALUE'] = df['FLAG'].apply(lambda x: 1 if x == 'B' else 0)
    return df


# Precipitation Data
def get_climate_data(stationID, year, timeframe):
    """Function to download precip data from Environment Canada climate stations for current reporting year"""
    url = f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={stationID}&Year={year}&Month=1&Day=14&timeframe={timeframe}&submit= Download+Data"
    response_csv = requests.get(url).text
    df = pd.read_csv(StringIO(response_csv), encoding='unicode_escape')
    return df


def get_rdpa_daily_data(year, month, day):
    """Function to download Regional Deterministic Precipitation Analysis (RDPA) grib2 files"""
    # Archived "nowcast"
    url = f"/vsicurl/https://collaboration.cmc.ec.gc.ca/science/outgoing/capa.grib/{year}{month}{day}12_000_CMC_RDPA_APCP-024-0700cutoff_SFC_0_ps10km.grib2"
    ds = gdal.Open(url)
    if ds is None:
        # Recent "nowcast"
        url = f"/vsicurl/https://dd.weather.gc.ca/analysis/precip/rdpa/grib2/polar_stereographic/24/CMC_RDPA_APCP-024-0700cutoff_SFC_0_ps10km_{year}{month}{day}12_000.grib2"
        ds = gdal.Open(url)

    if ds is None:
        # Archived "hindcast"
        url = f"/vsicurl/https://collaboration.cmc.ec.gc.ca/science/outgoing/capa.grib/hindcast/capa_hindcast_v2.4/24h/{year}{month}{day}12_000_CMC_RDPA_APCP-024_SFC_0_ps10km.grib2"
        ds = gdal.Open(url)
    # https://eccc-msc.github.io/open-data/msc-data/nwp_rdpa/readme_rdpa-datamart_en/
    # Band 0 (1) is Analysis of Accumulated Precipitation on a 06hr or 24hr interval
    # Band 1 (2) is Confidence Index for Analysis
    band = ds.GetRasterBand(1)
    dataset = band.ReadAsArray()
    dataset[dataset == band.GetNoDataValue()] = np.nan
    dataset = np.flip(dataset, 0)
    return dataset


def prepare_rdpa_data():
    """Function to format RDPA data into global dataframes"""
    global df_map_rdpa

    # RDPS/RDPA preformatted ASCII file https://meteo.gc.ca/grib/10km_res.bz2
    # https://eccc-msc.github.io/open-data/msc-data/nwp_rdps/readme_rdps-datamart_en/
    # dfcoord = pd.read_csv('10km_res')
    # coordinates of the first grid point should be 18.1429° N 142.8968° W
    # set starting index to 0,0
    # https://eccc-msc.github.io/open-data/msc-data/nwp_rdpa/readme_rdpa-datamart_en/
    # dfcoord['Longitude'] = dfcoord['Longitude'] - 360  # - 0.0043 + 0.000044
    # dfcoord['Latitude'] = dfcoord['Latitude']  # - 0.002130
    # dfcoord['ni'] = dfcoord['ni'] - 1
    # dfcoord['nj'] = dfcoord['nj'] - 1

    dt = pd.to_datetime(np.arange(f'{current_yr}-01-01', f'{int(current_yr) + 1}-01-01', dtype='datetime64[D]'))
    tot = np.zeros((len(dt), 824, 935))
    for cnt, i in enumerate(dt):
        try:
            ds = get_rdpa_daily_data(i.strftime('%Y'), i.strftime('%m'), i.strftime('%d'))
        except:  # All errors caught
            ds = np.nan
        tot[cnt, :, :] = ds

    stations = [*flow_stn, *level_stn]
    df_map_rdpa = []
    for sta in stations:
        shapefile = os.path.join(files_dir, 'rdpa_grids', f'CaPA_grid_{sta}.shp')
        if os.path.exists(shapefile):
            gdf = gpd.read_file(shapefile)
            gdf['Weight'] = gdf.area / np.sum(gdf.area)

            MAP = np.zeros((len(dt)))
            for i in range(0, len(gdf)):
                # -1 for nj and ni because the index in the shapefiles start at 1 instead of 0
                MAP += tot[:, gdf['nj'][i]-1, gdf['ni'][i]-1].squeeze() * gdf['Weight'][i]

            data = {'StationID': sta, 'DATE': dt, 'VALUE': MAP}
            df_map_rdpa.append(pd.DataFrame(data))

    df_map_rdpa = pd.concat(df_map_rdpa)

    return


def get_precip(stn):
    """Function to download precip data for each hydrometric station, based on the climate station specified in
    Station_Meta.csv"""
    CS_ID = df_meta[df_meta.ID == stn].CS.iloc[0]

    # Environment Canada Climate Stations
    ECCC = ['3060330', '3062697', '3064528']
    if CS_ID in ECCC:
        timeframe = 2  # 1 for hourly data, 2 for daily data, 3 for monthly data
        if CS_ID == '3062697':
            stationID = 49490
            CS = "Climate Station 3062697: Fort McMurray A (ECCC)"
        elif CS_ID == '3064528':
            stationID = 10978
            CS = "Climate Station 3064528: Mildred Lake (ECCC)"
        elif CS_ID == '3060330':
            stationID = 47047
            CS = "Climate Station 3060330: Athabasca AGCM (ECCC)"

        df = get_climate_data(stationID, current_yr, timeframe)
        df = df.filter(["Date/Time", "Total Precip (mm)"])
        df = df.rename(columns={"Date/Time": "DATE", "Total Precip (mm)": "VALUE"})

    # All other precip data must be saved locally
    else:
        df = df_precip[df_precip.CS_ID == CS_ID].reset_index()
        CS = f"Climate Station: {df.Name.iloc[0]} ({df.Source.iloc[0]})"

        df = df.rename(columns={"Date": "DATE", "Daily Precip": "VALUE"})
        df = df.filter(["DATE", "VALUE"])

    df["DATE"] = df["DATE"].astype('datetime64[ns]')
    return df, CS


def get_map_rdpa(stn):
    """Function to select RDPA MAP data for each hydrometric station"""
    df = df_map_rdpa[df_map_rdpa.StationID == stn].reset_index()
    df = df.filter(["DATE", "VALUE"])

    df["DATE"] = df["DATE"].astype('datetime64[ns]')
    return df


###################################################################
###                 Generate Annual Hydrographs                 ###
###################################################################
# Main Body: Key Indicator Stations
def plot(conn, stn):
    """Function to create hydrograph for key indicator stations
    (saves figures to folder in current directory)"""
    stn_name = get_stn_name(conn, stn)

    df = get_daily_flows(conn, stn)
    df_cy = df[(df['DATE'] > dt.datetime(previous_yr, 12, 31)) & (df['DATE'] < dt.datetime(next_yr, 1, 1))]
    # daily_flows = get_daily_flows_cy(conn, stn)
    flow = df_cy.FLOW
    f_date = df_cy.DATE

    # flow_quantiles = get_quantiles_flows(conn, stn)
    flow_quantiles = get_quantiles_flows(df)
    q_min = flow_quantiles[flow_quantiles.n >= 2].MIN
    q_max = flow_quantiles[flow_quantiles.n >= 2].MAX
    q_25 = flow_quantiles[flow_quantiles.n >= 5].p_25
    q_50 = flow_quantiles[flow_quantiles.n >= 5].p_50
    q_75 = flow_quantiles[flow_quantiles.n >= 5].p_75
    q_date = flow_quantiles.DATE

    # ice_period = get_ice_affected_period(conn, stn)
    ice_period = get_ice_affected_period(df_cy)
    ice = ice_period.VALUE
    ice_date = ice_period.DATE

    # ***************
    if not use_RDPA:
        precip, CS = get_precip(stn)
        p = precip.VALUE
        p_date = precip.DATE
    else:
        map_rdpa = get_map_rdpa(stn)
        if not map_rdpa.empty:
            p = map_rdpa.VALUE
            p_date = map_rdpa.DATE
    # ***************

    record = get_record_period_Q(conn, stn)
    RAMP_record = df_meta[df_meta.ID == stn]["RAMP Record"].iloc[0]

    # Create Figure
    fig, ax1 = plt.subplots(sharex=True)
    fig.set_size_inches(15, 8)
    fig.autofmt_xdate()  # rotates datetime on angle

    # Add Footnotes
    if df_meta[df_meta.ID == stn]["RAMP Record"].isnull().iloc[0]:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}*"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}*"

        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        max_label = f"Maximum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        min_label = f"Minimum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'*Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'*Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    else:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}**"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}**"

        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        max_label = f"Maximum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        min_label = f"Minimum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        plt.figtext(0.92, 0.03, f'*Monitoring occurred under RAMP from {RAMP_record}', ha='right', va='bottom')

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'**Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'**Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    # Daily Flows and Percentiles
    ax1.plot(f_date, flow, linewidth=2.5, label=f"Flow {current_yr}")

    ax1.fill_between(q_date, q_25, q_75, label=percentile_label, alpha=0.6, facecolor='yellow')
    ax1.plot(q_date, q_max, 'k', dashes=[5, 5, 5, 5], label=max_label)
    ax1.plot(q_date, q_min, 'k--', label=min_label)

    # Precipitation and Ice affected
    ax2 = ax1.twinx()
    ax2.bar(p_date, -p, label=precip_label, color='g', alpha=0.5)
    ax2.fill_between(ice_date, 0, -ice * 1000, label=f"Ice Affected {current_yr}", alpha=0.15)

    # Format x- and y-axes and gridlines
    ax1.set_ylabel('Discharge [m³/s]', fontsize=12)
    ax1.set_xlim(left=dt.datetime(current_yr, 1, 1), right=dt.datetime(current_yr, 12, 31))
    y_max = max([q_max.max(), flow.max()])
    ax1.set_ylim(0, (y_max * 1.2))

    ax2.set_ylabel('Precipitation [mm]', color='g', fontsize=12)
    y2_ticks = np.linspace(0, 100, 11)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(-100, 0)

    months = mdate.MonthLocator()
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%b-%d-%Y'))
    ax1.grid(which='both', alpha=0.5)

    # Format legend, title and save Figure
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0.68, 0.6), fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.savefig(os.path.join(files_dir, fr'{current_yr} Report\Figures\{stn_name} - {stn}.png'))
    plt.close()
    return


# Appendix C: Hydrographs for All Discharge Measurement Stations
def plot_appendix_flow(conn, stn):
    stn_name = get_stn_name(conn, stn)

    df = get_daily_flows(conn, stn)
    df_cy = df[(df['DATE'] > dt.datetime(previous_yr, 12, 31)) & (df['DATE'] < dt.datetime(next_yr, 1, 1))].copy()
    # daily_flows = get_daily_flows_cy(conn, stn)
    flow = df_cy.FLOW
    f_date = df_cy.DATE

    # flow_quantiles = get_quantiles_flows(conn, stn)
    flow_quantiles = get_quantiles_flows(df)
    q_min = flow_quantiles[flow_quantiles.n >= 2].MIN
    q_max = flow_quantiles[flow_quantiles.n >= 2].MAX
    q_25 = flow_quantiles[flow_quantiles.n >= 5].p_25
    q_50 = flow_quantiles[flow_quantiles.n >= 5].p_50
    q_75 = flow_quantiles[flow_quantiles.n >= 5].p_75
    q_date = flow_quantiles.DATE
    q_date_m = flow_quantiles[flow_quantiles.n >= 2].DATE
    q_date_q = flow_quantiles[flow_quantiles.n >= 5].DATE

    # ice_period = get_ice_affected_period(conn, stn)
    ice_period = get_ice_affected_period(df_cy)
    ice = ice_period.VALUE
    ice_date = ice_period.DATE

    # ***************
    if not use_RDPA:
        precip, CS = get_precip(stn)
        p = precip.VALUE
        p_date = precip.DATE
    else:
        map_rdpa = get_map_rdpa(stn)
        if not map_rdpa.empty:
            p = map_rdpa.VALUE
            p_date = map_rdpa.DATE
    # ***************

    record = get_record_period_Q(conn, stn)
    RAMP_record = df_meta[df_meta.ID == stn]["RAMP Record"].iloc[0]

    # Create Figure
    fig, ax1 = plt.subplots(sharex=True)
    fig.set_size_inches(15, 10)
    fig.autofmt_xdate()  # rotates datetime on angle

    # Add Footnotes
    if df_meta[df_meta.ID == stn]["RAMP Record"].isnull().iloc[0]:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}*"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}*"

        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        max_label = f"Maximum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        min_label = f"Minimum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'*Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'*Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    else:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}**"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}**"

        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        max_label = f"Maximum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        min_label = f"Minimum Flow on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        plt.figtext(0.92, 0.03, f'*Monitoring occurred under RAMP from {RAMP_record}', ha='right', va='bottom')

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'**Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'**Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    # Daily Flows and Percentiles
    ax1.plot(f_date, flow, linewidth=2.5, label=f"Flow {current_yr}")

    ax1.fill_between(q_date_q, q_25, q_75, label=percentile_label, alpha=0.6, facecolor='yellow')
    ax1.plot(q_date_m, q_max, 'k', dashes=[5, 5, 5, 5], label=max_label)
    ax1.plot(q_date_m, q_min, 'k--', label=min_label)

    # Precipitation and Ice affected
    ax2 = ax1.twinx()
    ax2.bar(p_date, -p, label=precip_label, color='g', alpha=0.5)
    ax2.fill_between(ice_date, 0, -ice * 1000, label=f"Ice Affected {current_yr}", alpha=0.15)

    # Format x- and y-axes and gridlines
    ax1.set_ylabel('Discharge [m³/s]', fontsize=12)
    ax1.set_xlim(left=dt.datetime(current_yr, 1, 1), right=dt.datetime(current_yr, 12, 31))
    y_max = max([q_max.max(), flow.max()])
    ax1.set_ylim(0, (y_max * 1.2))

    ax2.set_ylabel('Precipitation [mm]', color='g', fontsize=12)
    y2_ticks = np.linspace(0, 100, 11)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(-100, 0)

    months = mdate.MonthLocator()
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%b-%d-%Y'))
    ax1.grid(which='both', alpha=0.5)

    # Format legend, title and save Figure
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0.68, 0.6), fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.savefig(os.path.join(files_dir, fr'{current_yr} Report\Figures\Appendix\{stn_name} - {stn}.png'))
    plt.close()
    return


# Appendix C: Hydrographs for All Level Measurement Stations
def plot_appendix_level(conn, stn):
    stn_name = get_stn_name(conn, stn)

    df = get_daily_levels(conn, stn)
    df_cy = df[(df['DATE'] > dt.datetime(previous_yr, 12, 31)) & (df['DATE'] < dt.datetime(next_yr, 1, 1))].copy()
    df_cy.reset_index(level=None, drop=True, inplace=True, col_level=0)
    # daily_levels_cy = get_daily_levels_cy(conn, stn)
    level_cy = df_cy.LEVEL
    date_cy = df_cy.DATE
    min_date = date_cy.min()
    max_date = date_cy.max()

    df_ly = df[(df['DATE'] > dt.datetime(previous_yr - 1, 12, 31)) & (df['DATE'] < dt.datetime(current_yr, 1, 1))].copy()
    # daily_levels_ly = get_daily_levels_ly(conn, stn)
    df_ly.YEAR = df_ly.YEAR.replace(previous_yr, current_yr)
    df_ly.DATE = pd.to_datetime(df_ly[['YEAR', 'MONTH', 'DAY']], errors='coerce')
    level_ly = df_ly.LEVEL
    date_ly = df_ly.DATE

    # level_quantiles = get_quantiles_levels(conn, stn)
    level_quantiles = get_quantiles_levels(df)
    q_min = level_quantiles[(level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].MIN
    q_max = level_quantiles[(level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].MAX
    q_25 = level_quantiles[
        (level_quantiles.n >= 5) & (level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].p_25
    q_50 = level_quantiles[
        (level_quantiles.n >= 5) & (level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].p_50
    q_75 = level_quantiles[
        (level_quantiles.n >= 5) & (level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].p_75
    q_date = level_quantiles[(level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].DATE
    q_date_q = level_quantiles[(level_quantiles.n >= 5) &
                               (level_quantiles.DATE > min_date) & (level_quantiles.DATE < max_date)].DATE

    # ***************
    if not use_RDPA:
        precip, CS = get_precip(stn)
        p = precip.VALUE
        p_date = precip.DATE
    else:
        map_rdpa = get_map_rdpa(stn)
        if not map_rdpa.empty:
            p = map_rdpa.VALUE
            p_date = map_rdpa.DATE
    # ***************

    record = get_record_period_H(conn, stn)
    RAMP_record = df_meta[df_meta.ID == stn]["RAMP Record"].iloc[0]

    # Create Figure
    fig, ax1 = plt.subplots(sharex=True)
    fig.set_size_inches(15, 10)
    fig.autofmt_xdate()  # rotates datetime on angle

    # Add Footnotes
    if df_meta[df_meta.ID == stn]["RAMP Record"].isnull().iloc[0]:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}*"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}*"

        previous_label = f"Previous Year (Record {record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        max_label = f"Maximum Level on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"
        min_label = f"Minimum Level on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})"

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'*Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'*Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    else:

        if not use_RDPA:
            precip_label = f"Precipitation {current_yr}**"
        else:
            precip_label = f"Basin Mean Areal Precipitation {current_yr}**"

        previous_label = f"Previous Year (Record {record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        percentile_label = f"25th to 75th Percentiles ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        max_label = f"Maximum Level on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        min_label = f"Minimum Level on Record ({record.YEAR_FROM[0]} - {record.YEAR_TO[0]})*"
        plt.figtext(0.92, 0.03, f'*Monitoring occurred under RAMP from {RAMP_record}', ha='right', va='bottom')

        if not use_RDPA:
            plt.figtext(0.92, 0.01, f'**Precipitation from {CS}', ha='right', va='bottom')
        else:
            plt.figtext(0.92, 0.01, f'**Precipitation from Regional Deterministic Precipitation Analysis', ha='right',
                        va='bottom')

    # Daily Levels and Percentiles
    ax1.plot(date_cy, level_cy, linewidth=2.5, label=f"Level {current_yr}")

    # If not enough data to plot quantiles, plot the previous year of data    
    if level_quantiles.empty:
        ax1.plot(date_ly, level_ly, color='0.5', label=previous_label)
    else:
        ax1.fill_between(q_date_q, q_25, q_75, label=percentile_label, alpha=0.6, facecolor='yellow')
        ax1.plot(q_date, q_max, 'k', dashes=[5, 5, 5, 5], label=max_label)
        ax1.plot(q_date, q_min, 'k--', label=min_label)

    # Plot Precipitation on a separate axis
    ax2 = ax1.twinx()
    ax2.bar(p_date, -p, label=precip_label, color='g', alpha=0.5)

    # Format x- and y-axes and gridlines
    ax1.set_ylabel('Water Level [m]', fontsize=12)
    ax1.set_xlim(left=dt.datetime(current_yr, 1, 1), right=dt.datetime(current_yr, 12, 31))

    # Set water level y-axis limits based on min and max values
    try:
        y_min = q_min.min()
        y_max = max([q_max.max(), level_cy.max()])
        ax1.set_ylim(top=(y_max + (y_max - y_min) * 0.2))
    except:
        y_min = level_cy.min()
        y_max = level_cy.max()
        ax1.set_ylim(top=(y_max + (y_max - y_min) * 0.2))

    # Set precip y-axis limits based on max value
    ax2.set_ylabel('Precipitation [mm]', color='g', fontsize=12)
    y2_ticks = np.linspace(0, 100, 11)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(-100, 0)

    # Format Date Axis
    months = mdate.MonthLocator()
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%b-%d-%Y'))
    ax1.grid(which='both', alpha=0.5)

    # Format legend, title and save Figure
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.68, 0.9), fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.savefig(os.path.join(files_dir, fr'{current_yr} Report\Figures\Appendix\{stn_name} - {stn}.png'))
    plt.close()
    return


###################################################################
###                Generate Tables for Appendix B               ###
###################################################################
# Table B1: Discharge Measurement Stations
def get_discharge_station_details(conn, stn):
    pd.options.display.float_format = '{:,}'.format

    # Get data from HYDAT
    df = pd.read_sql_query(f"SELECT STATION_NUMBER, STATION_NAME FROM STATIONS WHERE STATION_NUMBER = '{stn}'", conn)

    # import record/station information (RAMP, HYDAT, Drainage Areas)
    DA = df_meta[df_meta.ID == stn].GDA.iloc[0]
    df['DRAINAGE_AREA_GROSS'] = DA

    record = get_record_period_Q(conn, stn)
    df['RECORD'] = record['YEAR_FROM'].astype(str) + ' - ' + record['YEAR_TO'].astype(str)

    # Get Average Discharge for Current Year and Long Term
    mean = get_mean_flows(conn, stn)
    try:
        cy_mean = round(mean[mean.YEAR == current_yr].MEAN.iloc[0].tolist(), sigfigs=3)
    except:
        try:
            cy_mean = round(mean[mean.YEAR == current_yr].MEAN.iloc[0], sigfigs=3)
        except:
            cy_mean = float("nan")

    try:
        lt_mean = round(mean.MEAN.mean().tolist(), sigfigs=3)
    except:
        try:
            lt_mean = round(mean.MEAN.mean(), sigfigs=3)
        except:
            lt_mean = float("nan")

    # Calculate Water Yield for Current Year and Long Term
    if DA > 0:
        try:
            cy_yield = round((cy_mean / DA * 31556952 / 1000).tolist(),
                             sigfigs=3)  # Gregorian Calendar Year (365.2425 days)
        except:
            cy_yield = float("nan")
        try:
            lt_yield = round((lt_mean / DA * 31556952 / 1000).tolist(),
                             sigfigs=3)  # Gregorian Calendar Year (365.2425 days)
        except:
            lt_yield = float("nan")
    else:
        cy_yield = float("nan")
        lt_yield = float("nan")

    # Calculate Current Year as Percentage of Historic Discharge        
    try:
        percent = round((cy_mean / lt_mean * 100), sigfigs=3)
    except:
        percent = float("nan")

    df['AVERAGE_FLOW'] = cy_mean
    df['YIELD'] = cy_yield
    df['LONG_TERM_AVERAGE_FLOW'] = lt_mean
    df['LONG_TERM_AVERAGE_YIELD'] = lt_yield
    df['PERCENT'] = percent

    # Rearrange Table
    cols = ['STATION_NAME', 'STATION_NUMBER', 'RECORD', 'DRAINAGE_AREA_GROSS', 'YIELD', 'LONG_TERM_AVERAGE_YIELD',
            'AVERAGE_FLOW', 'LONG_TERM_AVERAGE_FLOW', 'PERCENT']
    df = df[cols]

    # Formatting
    df = df.fillna('N/A')

    # Rename Columns
    df = df.rename(columns={'STATION_NAME': 'STATION',
                            'STATION_NUMBER': 'ID',
                            'AVERAGE_FLOW': f'{current_yr} MEAN DISCHARGE [m³/s]',
                            'LONG_TERM_AVERAGE_FLOW': 'PERIOD OF RECORD MEAN DISCHARGE [m³/s]',
                            'DRAINAGE_AREA_GROSS': 'GROSS DRAINAGE AREA [km²]',
                            'YIELD': f'YIELD: {current_yr} [mm/year]',
                            'LONG_TERM_AVERAGE_YIELD': 'AVERAGE YIELD: RECORD [mm/year]',
                            'PERCENT': f'{current_yr} DISCHARGE/ PERIOD OF RECORD AVERAGE ANNUAL DISCHARGE   [%]'})
    return df


# Table B2: Level Measurement Stations
def get_level_station_details(conn, stn):
    pd.options.display.float_format = '{:,}'.format

    # Get Station Info
    df = pd.read_sql_query(f"SELECT STATION_NUMBER, STATION_NAME FROM STATIONS WHERE STATION_NUMBER = '{stn}'", conn)

    # Record
    record = get_record_period_H(conn, stn)
    df['RECORD'] = record['YEAR_FROM'].astype(str) + ' - ' + record['YEAR_TO'].astype(str)

    # Get Average Level for Current Year and Long Term
    mean = get_mean_levels(conn, stn)
    try:
        cy_mean = round(mean[mean.YEAR == current_yr].MEAN.iloc[0].tolist(), decimals=3)
    except:
        try:
            cy_mean = round(mean[mean.YEAR == current_yr].MEAN.iloc[0], decimals=3)
        except:
            cy_mean = float("nan")

    try:
        lt_mean = round(mean.MEAN.mean().tolist(), decimals=3)
    except:
        try:
            lt_mean = round(mean.MEAN.mean(), decimals=3)
        except:
            lt_mean = float("nan")

    df['AVG'] = cy_mean
    df['AVG_LT'] = lt_mean

    # Current Year as Percentage of Historic Discharge
    df['DIFFERENCE'] = df.AVG - df.AVG_LT
    df['DIFFERENCE'] = df[df['DIFFERENCE'].notnull()]['DIFFERENCE'].apply(lambda x: round(x, decimals=3))

    # Rearrange Table
    cols = ['STATION_NAME', 'STATION_NUMBER', 'RECORD', 'AVG', 'AVG_LT', 'DIFFERENCE']
    df = df[cols]

    # Formatting
    df = df.fillna('N/A')

    # Rename Columns
    df = df.rename(columns={'STATION_NAME': 'STATION',
                            'STATION_NUMBER': 'ID',
                            'AVG': f'{current_yr} MEAN LEVEL [m]',
                            'AVG_LT': 'HISTORICAL MEAN ANNUAL LEVEL [m]',
                            'DIFFERENCE': 'DIFFERENCE [m]'})
    return df


###################################################################
###                  Script to run the program                  ###
###################################################################
def Run():
    # Define variables globally to start, then overwrite them when running
    global current_yr
    global previous_yr
    global next_yr
    if not use_RDPA:
        global df_precip

    # Specify reporting year
    print("Enter reporting year ")
    x = input()

    # Set current reporting period
    current_yr = int(x)
    previous_yr = current_yr - 1
    next_yr = current_yr + 1

    if not use_RDPA:
        # Daily Precip Data
        df_precip = pd.read_csv(os.path.join(files_dir, f'DailyPrecip{current_yr}.csv'), low_memory=False)

    # Create station folder (all files to be saved here)
    annual_report_path = os.path.join(files_dir, f'{x} Report')
    figures_path = os.path.join(annual_report_path, 'Figures')
    appendix_figures_path = os.path.join(figures_path, 'Appendix')

    if not os.path.exists(annual_report_path):
        os.mkdir(annual_report_path)
        os.mkdir(figures_path)
        os.mkdir(appendix_figures_path)
        print(f"Directory '{annual_report_path}' created")

    if use_RDPA:
        start_timer = timer()
        prepare_rdpa_data()
        print(f'prepare_rdpa_data() run in {timer() - start_timer} seconds.')

    start_timer = timer()
    # Generate figures for main body of report: Key Indicator Stations
    for i in stn:
        plot(conn, i)
    print(f'plot() run in {timer() - start_timer} seconds.')

    start_timer = timer()
    # Generate flow hydrographs for all discharge stations (in alphabetical order)
    for i in flow_stn:
        plot_appendix_flow(conn, i)
    print(f'plot_appendix_flow() run in {timer() - start_timer} seconds.')

    start_timer = timer()
    # Generate flow hydrographs for all level stations (in alphabetical order)
    for i in level_stn:
        plot_appendix_level(conn, i)
    print(f'plot_appendix_level() run in {timer() - start_timer} seconds.')

    start_timer = timer()
    # Table B1
    frames_B1 = [get_discharge_station_details(conn, stn) for stn in flow_stn]
    Appendix_B1 = pd.concat(frames_B1)
    Appendix_B1 = Appendix_B1.reset_index(drop=True)
    Appendix_B1.to_csv((os.path.join(files_dir, f'{current_yr} Report', 'Appendix_B1.csv')), index=False, header=True)
    print(f'Appendix B1 created in {timer() - start_timer} seconds.')

    start_timer = timer()
    # Table B2
    frames_B2 = [get_level_station_details(conn, stn) for stn in level_stn]
    Appendix_B2 = pd.concat(frames_B2)
    Appendix_B2 = Appendix_B2.reset_index(drop=True)
    Appendix_B2.to_csv((os.path.join(files_dir, f'{current_yr} Report', 'Appendix_B2.csv')), index=False, header=True)
    print(f'Appendix B2 created in {timer() - start_timer} seconds.')

    return


# Run the program
if __name__ == '__main__':
    Run()
