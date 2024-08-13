"""umlaut CLI for visualizing metrics

The visualization frontend of umlaut.
Given a database file as created by an umlaut.Benchmark object,
this tool gives a selection of available metrics and helps users choose a subset to visualize.
Users can select metrics in two ways:
    1. by using command line arguments (see umlaut-cli -h)
    2. If no arguments are provided, umlaut-cli will prompt the user for arguments.
All metrics of the same measurement type are then visualized in a single diagram.
"""

import argparse
from itertools import chain
import pickle
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from PyInquirer import prompt, Separator
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from umlaut import VisualizationBenchmark
from umlaut.datamodel import Measurement, BenchmarkMetadata
from umlaut.visualization import type_to_visualizer_class_mapper

def get_args():
    file_help = "Sqlite Database file as created by an umlaut.Benchmark object"
    module_help = "Visualization CLI for End to End ML System Benchmark"
    uuid_help = "UUUIDs of the runs to visualize. Each uuid corresponds to one pipeline run. " + \
                "For humans uuids are usually tedious to handle. Leave this parameter out to be shown a list of available uuids to choose from."
    type_help = "Measurement types of the metrics to visualize. Each metric that umlaut supports has a descriptive type. " + \
                "Like the uuids, this parameter is optional and can be choosen by prompt."
    description_help = "Descriptions are supplied by users during the implementation stage of a pipeline. " + \
                       "They help giving descriptive information about captured metrics. " + \
                       "This parameter is optional and can be choosen via prompt later."
    plotting_backend_help = "Plotting backend used for visualization. The default is matplotlib."


    parser = argparse.ArgumentParser(description=module_help)
    parser.add_argument("file", help=file_help)
    parser.add_argument("-u", "--uuids", nargs="+", help=uuid_help, required=False)
    parser.add_argument("-t", "--types", nargs="+", help=type_help, required=False)
    parser.add_argument("-d", "--descriptions", nargs="+", help=description_help, required=False)
    parser.add_argument("-p", "--plotting-backend", choices=["matplotlib", "plotly", "text"], default="plotly", help=plotting_backend_help)
    
    return parser.parse_args()


def filter_by_args(meas_df, meta_df, args):

    if args.uuids is not  None:
        meas_df = meas_df[meas_df['uuid'].isin(args.uuids)]
        meta_df = meta_df[meta_df.index.isin(args.uuids)]
    if args.types is not None:
        meas_df = meas_df[meas_df['measurement_type'].isin(args.types)]
    if args.descriptions is not None:
        meas_df = meas_df[meas_df['measurement_description'].isin(args.descriptions)]

    if meas_df.empty:
        raise Exception("There are no database entries with the given uuids, types and descriptions.")

    return meas_df, meta_df

def prompt_for_uuid(meas_df, meta_df):
    prompt_choices = [
            {'name' : 
                f"{uuid}" +
                f", {str(db_entry['meta_start_time'])}" +
                f"{', description: ' + db_entry['meta_description'] if db_entry['meta_description'] else ''}",
             'value': uuid}
            for uuid, db_entry in meta_df.iterrows()
    ]
    uuid_prompt = {
        'type' : 'checkbox',
        'name' : 'uuids',
        'message' : 'Please select one or more uuids.',
        'choices' : prompt_choices
    }
    chosen_uuids = prompt(uuid_prompt)['uuids']
    meas_df = meas_df[meas_df['uuid'].isin(chosen_uuids)]
    meta_df = meta_df[meta_df.index.isin(chosen_uuids)]

    return meas_df, meta_df

def prompt_for_types(meas_df):
    prompt_choices = []
    for uuid, uuid_group in meas_df.groupby('uuid'):
        prompt_choices.append(Separator(f"Available types for uuid {uuid}"))
        for meas_type, meas_type_group in uuid_group.groupby('measurement_type'):
            prompt_choices.append(
                {'name' : meas_type,
                 'value' : list(meas_type_group.index)}
            )
    type_prompt = {
        'type' : 'checkbox',
        'name' : 'measurement_types',
        'message' : 'Please select measurement types corresponding to uuids.',
        'choices' : prompt_choices
    }
    prompt_res = prompt(type_prompt)['measurement_types']
    idx = list(chain(*prompt_res))
    meas_df = meas_df[meas_df.index.isin(idx)]
    
    return meas_df

def prompt_for_description(meas_df):
    prompt_choices = []
    for (uuid, meas_type), u_t_group in meas_df.groupby(['uuid', 'measurement_type']):
        prompt_choices.append(Separator(f"Available descriptions for uuid {uuid} and type {meas_type}"))
        for desc, desc_group in u_t_group.groupby('measurement_description'):
            prompt_choices.append(
                {'name' : desc,
                 'value' : list(desc_group.index)}
            )
    desc_prompt = {
        'type' : 'checkbox',
        'name' : 'descriptions',
        'message' : 'Please select descriptions corresponding to uuids and types.',
        'choices' : prompt_choices
    }
    prompt_res = prompt(desc_prompt)['descriptions']
    idx = list(chain(*prompt_res))
    meas_df = meas_df[meas_df.index.isin(idx)]
    
    return meas_df

def main():
    args = get_args()
    benchmark = VisualizationBenchmark(args.file)
    meas_df = benchmark.query_all_uuid_type_desc()
    meta_df = benchmark.query_all_meta()
    meas_df, meta_df = filter_by_args(meas_df, meta_df, args)
    
    if args.uuids is None:
        meas_df, meta_df = prompt_for_uuid(meas_df, meta_df)
    if args.types is None:
        meas_df = prompt_for_types(meas_df)
    if args.descriptions is None:
        meas_df = prompt_for_description(meas_df)
    
    df = benchmark.join_visualization_queries(meas_df)
    
    figs = []

    for meas_type, type_group_df in df.groupby('measurement_type'):
        VisualizerClass = type_to_visualizer_class_mapper[meas_type]
        visualizer = VisualizerClass(type_group_df, args.plotting_backend)
        figs.append(visualizer.plot())

    if args.plotting_backend == 'matplotlib':
        figs = list(chain(*figs))
        plt.show()
    if args.plotting_backend == 'plotly':
        figs = list(chain(*figs))
        for fig in figs:
            fig.show()

    benchmark.close()


if __name__ == "__main__":
    main()
