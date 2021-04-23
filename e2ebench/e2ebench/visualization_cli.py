import argparse
from collections import defaultdict
from itertools import chain
from operator import attrgetter
import os
import sys

import pandas as pd
from PyInquirer import prompt, Separator
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from e2ebench.datamodel import Measurement
from e2ebench.datamodel import BenchmarkMetadata
from e2ebench import metrics

def get_args():
    parser = argparse.ArgumentParser(description="Visualization CLI for End to End ML System Benchmark")
    parser.add_argument("file", help="Sqlite Database file created by the benchmark package")
    parser.add_argument("-u", "--uuids", nargs="+", help="UUIDs of the benchmarks to visualize", required=False)
    parser.add_argument("-t", "--types", nargs="+", help="measurement types", required=False)
    parser.add_argument("-d", "--descriptions", nargs="+", help="descriptions", required=False)
    
    return parser.parse_args()

def create_dfs(session):
    meas_query = session.query(Measurement.id,
                               Measurement.benchmark_uuid,
                               Measurement.measurement_type,
                               Measurement.description)
    meta_query = session.query(BenchmarkMetadata.uuid,
                               BenchmarkMetadata.description,
                               BenchmarkMetadata.start_time)
    meas_df = pd.DataFrame(meas_query.all(), columns=['id', 'uuid', 'type', 'desc'])
    meta_df = pd.DataFrame(meta_query.all(), columns=['uuid', 'desc', 'start_time'])

    return meas_df, meta_df

def filter_by_args(meas_df, meta_df, args):
    if args.uuids:
        meas_df = meas_df[meas_df['uuid'].isin([args.uuids])]
        meta_df = meta_df[meta_df['uuid'].isin([args.uuids])]
    if args.types:
        meas_df = meas_df[meas_df['type'].isin([args.types])]
    if args.descriptions:
        meas_df = meas_df[meas_df['desc'].isin([args.descriptions])]

    if meas_df.empty:
        raise Exception("There are no database entries with the given uuids, types and descriptions.")

    return meas_df, meta_df

def prompt_for_uuid(meas_df, meta_df):
    prompt_choices = [
            {'name' : 
                f"{db_entry['uuid']}" +
                f", {str(db_entry['start_time'])}" +
                f"{', description: ' + db_entry['desc'] if db_entry['desc'] else ''}",
             'value': db_entry['uuid']}
            for _, db_entry in meta_df.iterrows()
    ]
    uuid_prompt = {
        'type' : 'checkbox',
        'name' : 'uuids',
        'message' : 'Please select one or more uuids.',
        'choices' : prompt_choices
    }
    chosen_uuids = prompt(uuid_prompt)['uuids']
    meas_df = meas_df[meas_df['uuid'].isin(chosen_uuids)]
    meta_df = meta_df[meta_df['uuid'].isin(chosen_uuids)]

    return meas_df, meta_df

def prompt_for_types(meas_df):
    prompt_choices = []
    for uuid, uuid_group in meas_df.groupby('uuid'):
        prompt_choices.append(Separator(f"Available types for uuid {uuid}"))
        for meas_type, meas_type_group in uuid_group.groupby('type'):
            prompt_choices.append(
                {'name' : meas_type,
                 'value' : list(meas_type_group['id'])}
            )
    type_prompt = {
        'type' : 'checkbox',
        'name' : 'measurement_types',
        'message' : 'Please select measurement types corresponding to uuids.',
        'choices' : prompt_choices
    }
    prompt_res = prompt(type_prompt)['measurement_types']
    idx = list(chain(*prompt_res))
    meas_df = meas_df[meas_df['id'].isin(idx)]
    
    return meas_df

def prompt_for_description(meas_df):
    prompt_choices = []
    for (uuid, meas_type), u_t_group in meas_df.groupby(['uuid', 'type']):
        prompt_choices.append(Separator(f"Available descriptions for uuid {uuid} and type {meas_type}"))
        for desc, desc_group in u_t_group.groupby('desc'):
            prompt_choices.append(
                {'name' : desc,
                    'value' : list(desc_group['id'])}
            )
    desc_prompt = {
        'type' : 'checkbox',
        'name' : 'descriptions',
        'message' : 'Please select descriptions corresponding to uuids and types.',
        'choices' : prompt_choices
    }
    prompt_res = prompt(desc_prompt)['descriptions']
    idx = list(chain(*prompt_res))
    meas_df = meas_df[meas_df['id'].isin(idx)]
    
    return meas_df



def main():
    args = get_args()

    engine = create_engine(f'sqlite+pysqlite:///{args.file}')
    Session = sessionmaker(bind=engine)
    session = Session()

    meas_df, meta_df = create_dfs(session)
    meas_df, meta_df = filter_by_args(meas_df, meta_df, args)
    
    if args.uuids is None:
        meas_df, meta_df = prompt_for_uuid(meas_df, meta_df)
    if args.types is None:
        meas_df = prompt_for_types(meas_df)
    if args.descriptions is None:
        meas_df = prompt_for_description(meas_df)
    
    

    

if __name__ == "__main__":
    main()
