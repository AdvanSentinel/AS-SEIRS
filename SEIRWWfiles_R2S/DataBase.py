#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import gmean
NUM_OF_ROLL = 3

def get_dataframe(prd_meas_wide_analyzed, prd_geom_polygon, prd_public_health_data_analyzed,geom_id_idx,geom_id, ww_idx, assay_method="HS",ww_aggr="mean"):
    # Merge geom_polygon Id
    sel_cols = ['site_id','geom_id_0', 'geom_id_1', 'geom_id_2']
    df_merged = pd.merge(prd_meas_wide_analyzed,prd_geom_polygon[sel_cols], on="site_id", how="left")

    # Select data from database
    def select_data_from_geom_id(df, df_phd, geom_id_idx, geom_id, assay_method):
        df_tmp = df[df[geom_id_idx]==geom_id].copy()
        df_tmp = df_tmp[df_tmp["assay_method"] == assay_method].copy()
        # original value ND=-1, BLQ = 0
        #df_tmp["covn_1_use01"] = np.where(df_tmp["rmk_covn_1"]=="ND",0, np.where(df_tmp["rmk_covn_1"]=="BLQ",231.5, df_tmp["covn_1_usemeas"]))
        #df_tmp["covn_1_use01"] = np.where(df_tmp["rmk_covn_1"]=="ND",0, df_tmp["covn_1_usemeas"])
        df_tmp["covn_1_use01"] = np.where(df_tmp["rmk_covn_1"]=="ND",1, df_tmp["covn_1_usemeas"])
        df_tmp["log_covn_1_tbl"] = np.where(df_tmp["rmk_covn_1"]=="ND",1, df_tmp["covn_1_tbl"])
        df_tmp["log_covn_1_tbl"] = np.log10(df_tmp["log_covn_1_tbl"])
        df_tmp["log_covn_1_use01"] = np.where(df_tmp["rmk_covn_1"]=="ND",1, np.where(df_tmp["rmk_covn_1"]=="BLQ",1, df_tmp["covn_1_usemeas"]))
        df_tmp["log_covn_1_use01"] = np.log10(df_tmp["log_covn_1_use01"])
        df_tmp["covn_1_to_pmmov_user01"] = df_tmp["covn_1_use01"]/df_tmp["pmmov_tbl"]*1e4

        # time type 処理 and merge phd data
        df_tmp["dateTime"] = pd.to_datetime(df_tmp["sampling_date"],format="%Y/%m/%d")

        sel_cols=["covn_1_usemeas","covn_1_uselloq","covn_1_tbl","covn_1_use01","covn_1_to_pmmov_user01"]
        mean_cols = sel_cols
        # Do aggregation first
        if ww_aggr == "mean":
            def aggregator(x):
                vals = x[mean_cols].mean()
                return pd.Series({"site_name":",".join(x["site_name"]),"rmk_covn_1":",".join(x["rmk_covn_1"]),**vals})
            df_tmp = df_tmp.groupby([geom_id_idx,'dateTime','sampling_date']).apply(aggregator).reset_index()
        elif ww_aggr == "max":
            def aggregator(x):
                vals = x[mean_cols].max()
                return pd.Series({"site_name":",".join(x["site_name"]),"rmk_covn_1":",".join(x["rmk_covn_1"]),**vals})
            df_tmp = df_tmp.groupby([geom_id_idx,'dateTime','sampling_date']).apply(aggregator).reset_index()
        elif ww_aggr == "gmean":
            def aggregator(x):
                vals = x[mean_cols].apply(gmean)
                return pd.Series({"site_name":",".join(x["site_name"]),"rmk_covn_1":",".join(x["rmk_covn_1"]),**vals})
            df_tmp = df_tmp.groupby([geom_id_idx,'dateTime','sampling_date']).apply(aggregator).reset_index()
        else:
            raise Exception("Error no such aggregation:",ww_aggr)
        df_tmp["aggr"] = ww_aggr
        #debug print(df_tmp.columns)
        # print(df_tmp)
        
        # Do rolling mean
        for i in sel_cols:
            #print(i,df_tmp[["site_name","assay_method","sampling_date",i]])
            df_tmp["rol"+str(NUM_OF_ROLL)+"_"+i] = df_tmp[i].rolling(window=NUM_OF_ROLL, min_periods=1).mean()
            row_numbers = df_tmp["rol"+str(NUM_OF_ROLL)+"_"+i].index

        # Select phd data
        df_phd = df_phd[df_phd[geom_id_idx]==geom_id].copy()
        df_phd["dateTime"] = pd.to_datetime(df_phd["phd_date"],format="%Y/%m/%d")

        sel_cols = ['site_id','val_conf_report','val_7_d_conf_report','dateTime',geom_id_idx]

        df_tmp = pd.merge(df_phd[sel_cols], df_tmp, on=["dateTime",geom_id_idx],how="left")
        return df_tmp
     
    if assay_method == "HS":
        df_merged = select_data_from_geom_id(df_merged, prd_public_health_data_analyzed, geom_id_idx, geom_id, "HS")
    elif assay_method == "CM":
        df_merged = select_data_from_geom_id(df_merged, prd_public_health_data_analyzed, geom_id_idx, geom_id, "CM")
    else:
        print("ERROR unknown assay method")
        return 

    # Set Case and WW
    df_merged["ww"]    = df_merged[ww_idx].copy()
    df_merged["cases"] = df_merged["val_conf_report"].copy()
    df_merged["date"]  = df_merged["dateTime"].copy()
    sel_cols=["site_name","dateTime","sampling_date","covn_1_usemeas","rol"+str(NUM_OF_ROLL)+"_covn_1_usemeas","covn_1_uselloq","covn_1_tbl","covn_1_use01","rmk_covn_1","covn_1_to_pmmov_user01","ww","cases"]
    df_tmp=df_merged[sel_cols].copy()
    df_tmp = df_tmp.dropna()
    list_set_zero_roll = df_tmp.index[0:2]
    
    df_rtn = df_merged.copy()
    # Reset zero 
    sel_cols=["covn_1_usemeas","covn_1_uselloq","covn_1_tbl","covn_1_use01"]
    for i in sel_cols:
        idx = "rol"+str(NUM_OF_ROLL)+"_"+i
        for j in list_set_zero_roll:
            df_rtn[idx].iloc[j] = 0


     
    # Drop data
    #df_merged = df_merged[410:len(df_merged)].reset_index()
    #df_merged = df_merged[0:220].reset_index()
    sel_cols=["site_name","dateTime","sampling_date","covn_1_usemeas","rol"+str(NUM_OF_ROLL)+"_covn_1_usemeas","covn_1_uselloq","covn_1_tbl","covn_1_use01","rmk_covn_1","covn_1_to_pmmov_user01","ww","cases"]
    df_tmp = df_merged[sel_cols]
    df_tmp.to_csv("check_xlsx.csv", header=True,encoding="shift_jis")

    # Fill nan        
    df_rtn = df_rtn.fillna(-1)
    #print(df_tmp.head(20))
    return df_rtn
