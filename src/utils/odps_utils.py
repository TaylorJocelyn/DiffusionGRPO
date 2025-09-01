#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 11:04
# @Author  : 晨皋
# @File    : odps_utils.py.py
# @Software: PyCharm
from odps.inter import enter
from odps.df import DataFrame
import os
import pandas as pd
import common_io

def get_cache_filepath(table_name, odps_df, cache_df_dir):
    row_num = odps_df.count().execute()
    if not os.path.exists(cache_df_dir):
        os.makedirs(cache_df_dir)
    save_name = "{}_{}".format(table_name, row_num)
    file_path = os.path.join(cache_df_dir, f'{save_name}.csv')
    return file_path


def get_df_from_odps_table(o, table_name, selected_cols=None, use_cache=False):
    if selected_cols is None:
        odps_df = DataFrame(o.get_table(table_name)).to_pandas()
    else:
        odps_df = DataFrame(o.get_table(table_name))[selected_cols]
    cache_filepath = get_cache_filepath(table_name, odps_df)

    if use_cache and os.path.isfile(cache_filepath):
        df = pd.read_csv(cache_filepath)
        print(f"read from cache file '{cache_filepath}'")
        return df

    if selected_cols is None:
        df = DataFrame(o.get_table(table_name)).to_pandas()
    else:
        df = DataFrame(o.get_table(table_name))[selected_cols].to_pandas()
    df.to_csv(cache_filepath, index=False)
    print("download {} done, save to cache file path {}".format(table_name, cache_filepath))
    return df


def convert_to_df_and_upload_to_odps(dataset, sql_table_name, multi_label=False):
    data = []
    for batch_row_idx, value in enumerate(dataset):
        input_ids_list = ",".join(map(str, value["input_ids"]))
        attention_mask_list = ",".join(map(str, value["attention_mask"]))
        if multi_label:
            labels = ",".join(map(str, value["labels"]))
        else:
            labels = value["labels"]
        data.append([input_ids_list, attention_mask_list, labels])
    # 将提取的数据转换为DataFrame
    df = pd.DataFrame(data, columns=["input_ids", "attention_mask", "labels"])
    pdf = DataFrame(df)
    o.delete_table(sql_table_name, if_exists=True)
    pdf.persist(sql_table_name, lifecycle=180)
    print("uplaod to {} done".format(sql_table_name))

def read_odps_table(table_path, selected_cols="", slice_id=0, slice_count=1):
    reader = common_io.table.TableReader(table_path,
                                         selected_cols=selected_cols,
                                         slice_id=slice_id,
                                         slice_count=slice_count)
    records_cnt = reader.get_row_count()
    records = reader.read(records_cnt, allow_smaller_final_batch=True)
    reader.close()
    return records