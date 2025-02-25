from datetime import datetime
import os
import shutil
import pickle
from queue import Queue
from threading import Thread, Event
from time import sleep
from uuid import uuid4
import json
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .datamodel import Base, Measurement, BenchmarkMetadata
import inspect

# Util for the benchmark name
def get_outermost_filename():
    current_frame = inspect.currentframe()
    
    # Traverse to the outermost frame
    outermost_frame = current_frame
    while current_frame:
        outermost_frame = current_frame
        current_frame = current_frame.f_back
    
    # Get the file name of the outermost frame
    filename = outermost_frame.f_globals.get("__file__", None)
    
    if filename:
        return os.path.basename(filename)
    return ""

class Benchmark:
    """A class that manages the database entries for the measured metrics which are logged into the database.

    Parameters
    ----------
    db_file : str
        The path of the database file
    description : str, optional
        The description of the whole pipeline use case. Even though the description is optional, it should be set
        so the database entries are distinguishable without evaluating the uuid's.
        This parameter is ignored for Benchmark objects initialized in mode 'r'.
    name : str, optional
        The name of the current pipeline run. It is the file name by default.
    mode : str, default='a'
        One of ['w', 'a', 'r']. The mode corresponds to conventional python file handling modes.
        Modes 'a' and 'w' are used for storing metrics in a database during a pipeline run
        and 'r' is used for querying metrics from the database.

    Attributes
    ----------
    db_file : str
        path to the database file
    mode : str
        mode of the Benchmark object. One of ['w', 'a', 'r'].
    description : str
        description of the pipeline run. Not relevant if mode is 'r'.
    name : str
        name of the current pipeline run.
    session : sqlalchemy.orm.session.Session
        SQLalchemy session
    """

    def __init__(self, db_file, description="", mode="a", name="", checkpoint_frequency=2, tmp_dir="umlaut_tmp"):
        self.db_file = db_file
        self.description = description
        self.name = name if name != "" else get_outermost_filename()
        self.mode = mode
        self.checkpoint_frequency = checkpoint_frequency
        self.log_idx = 0
        self.tmp_dir = tmp_dir
        
        # Ensure temporary directory exists
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        if mode == 'r':
            if not os.path.exists(self.db_file):
                raise FileNotFoundError("Cannot open a non-existing file in reading mode.")
            engine = create_engine('sqlite+pysqlite:///' + self.db_file)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.session = Session()

        if mode == 'w':
            if os.path.exists(self.db_file):
                os.remove(self.db_file)
        
        if mode in ['w', 'a']:
            self.close_event = Event()
            self.uuid = str(uuid4())
            self.queue = Queue()
            
            engine = create_engine('sqlite+pysqlite:///' + self.db_file)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.session = Session()
            # self.session.add(BenchmarkMetadata(uuid=self.uuid,
            #                             meta_description=self.description,
            #                             meta_name=self.name,
            #                             meta_start_time=datetime.now()))
            # self.session.commit()
            temp_file_path = os.path.join(self.tmp_dir, "metadata.json")
            with open(temp_file_path, 'w') as temp_file:
                print({"uuid": self.uuid, "meta_description": self.description, "meta_name": self.name, "meta_start_time": datetime.now().isoformat()})
                json.dump({"uuid": self.uuid, "meta_description": self.description, "meta_name": self.name, "meta_start_time": datetime.now().isoformat()}, temp_file)

    def query(self, *args, **kwargs):
        """
        Send queries to the database file.
        You can send queries in the same manner you would query an SQLalchemy session.

        This method only works in mode 'r'.
        """
        if self.mode != "r":
            raise Exception("Invalid file mode. Mode must be \"r\" to send queries.")

        return self.session.query(*args, **kwargs)
    
    def read_checkpoints(self):
        metadata_file_path = os.path.join(self.tmp_dir, "metadata.json")
        with open(metadata_file_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
            metadata["meta_start_time"] = datetime.fromisoformat(metadata["meta_start_time"])
        
        measurements = []
        for filename in os.listdir(self.tmp_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.tmp_dir, filename)
                with open(file_path, 'rb') as temp_file:
                    measurement = pickle.load(temp_file)
                    measurements.append(measurement)
            
        groups = []    
        for identifier in set([(measurement.measurement_type, measurement.measurement_description, measurement.measured_method_name) for measurement in measurements]):
            print(identifier)
            groups.append([measurement for measurement in measurements if (measurement.measurement_type, measurement.measurement_description, measurement.measured_method_name) == identifier])
            
        processed_measurements = []
        for group in groups:
            if len(group) == 0:
                continue
            elif len(group) == 1:
                processed_measurements.append(group[0])
            else:
                # Combine measurements
                combined_measurement = group[0]
                value = combined_measurement.measurement_data
                if type(value) in [int, float]:
                    for measurement in group[1:]:
                        value += measurement.measurement_data
                elif type(value) == str:
                    value = json.loads(value)
                    for measurement in group[1:]:
                        next_value = json.loads(measurement.measurement_data)
                        for key in next_value:
                            if key in value:
                                value[key] += next_value[key]
                            else:
                                value[key] = next_value[key]
                                
                    for key in value:
                        if key != "timestamps":
                            value[key] = [item[1] for item in sorted(list(zip(value["timestamps"], value[key])), key = lambda x: x[0])]

                    value["timestamps"] = sorted(value["timestamps"])
                    value = json.dumps(value, indent=4, default=str)
                else:
                    raise Exception(f"Cannot combine measurements of type {type(value)}.")
                combined_measurement.measurement_data = value
                combined_measurement.measurement_datetime = datetime.now()
                processed_measurements.append(combined_measurement)
                
        return processed_measurements, metadata
    
    def write_checkpoints(self):
        """
        Write all measurements to the database.
        """
        measurements, metadata = self.read_checkpoints()
        self.session.add(BenchmarkMetadata(uuid=metadata["uuid"],
                                           meta_description=metadata["meta_description"],
                                           meta_name=metadata["meta_name"],
                                           meta_start_time=metadata["meta_start_time"]))
        
        for measurement in measurements:
            self.session.add(measurement)

        self.session.commit()
        print(f"All measurements committed to database.")

    def close(self):
        """
        Close the Benchmark object.
        All temporary JSON files are read, their data is written to the database, and the temporary directory is deleted.
        """
        print(f"Closing Benchmark. Processing {self.log_idx} logged measurements...")
        if self.mode == 'r':
            self.session.close()
        else:
            try:
                self.write_checkpoints()
                print(f"All measurements committed to database.")
            except Exception as e:
                print(f"Error during closing: {e}")
            finally:
                # Clean up temporary files and directory
                shutil.rmtree(self.tmp_dir, ignore_errors=True)
                if os.path.exists(self.tmp_dir):
                    os.rmdir(self.tmp_dir, ignore_errors=True)
                print(f"Temporary directory '{self.tmp_dir}' deleted.")
                self.session.close()
            

    def log(self, description, measure_type, value, unit='', method_name=""):
        """
        Logs a measurement into a temporary Pickle file in the tmp directory.
        """
        measurement = Measurement(
            measurement_datetime=datetime.now(),
            uuid=self.uuid,
            measurement_description=description,
            measurement_type=measure_type,
            measurement_data=value,
            measurement_unit=unit,
            measured_method_name=method_name
        )

        temp_file_path = os.path.join(self.tmp_dir, f"log_{self.log_idx}.pkl")
        with open(temp_file_path, 'wb') as temp_file:
            pickle.dump(measurement, temp_file)

        self.log_idx += 1
        print(f"Logged measurement {self.log_idx} to temporary file: {temp_file_path}")
  

class VisualizationBenchmark(Benchmark):
    def __init__(self, db_file, tmp_dir=None):
        if tmp_dir is None:
            tmp_dir = "umlaut_tmp"
        super().__init__(db_file, mode='r', tmp_dir=tmp_dir)

    def query_all_meta(self):
        """
        Returns a dataframe of all entries in BenchmarkMetadata and sets uuid as the index.
        """
        query = self.query(BenchmarkMetadata.uuid,
                           BenchmarkMetadata.meta_description,
                           BenchmarkMetadata.meta_name,
                           BenchmarkMetadata.meta_start_time)
        col_names = [col_desc['name'] for col_desc in query.column_descriptions]

        return pd.DataFrame(query.all(), columns=col_names).set_index('uuid')

    def query_all_uuid_type_desc(self):
        """
        Returns a dataframe of all entries in Measurement and sets Measurement.id as the index.
        """      
        query = self.query(Measurement.id,
                           Measurement.uuid,
                           Measurement.measurement_type,
                           Measurement.measurement_description)
        col_names = [col_desc['name'] for col_desc in query.column_descriptions]

        return pd.DataFrame(query.all(), columns=col_names).set_index('id')

    def join_visualization_queries(self, uuid_type_desc_df):
        """
        Joins all remaining database columns from both tables to uuid_type_desc_df.

        Parameters
        ----------
        uuid_type_desc_df : pandas.DataFrame
            Dataframe containing the columns uuid, measurement_type, measurement_description
            and Measurement.id as the index
            (same schema as returned by VisualizationBenchmark.query_all_uuid_type_desc()).

        Returns
        -------
        joined_df : pandas.DataFrame
            uuid_type_desc_df joined with the remaining database columns from both tables.
            Measurement.id is still the index in joined_df.
        """
        meta_query = self.query(BenchmarkMetadata.uuid,
                                BenchmarkMetadata.meta_start_time,
                                BenchmarkMetadata.meta_description,
                                BenchmarkMetadata.meta_name).filter(
                                    BenchmarkMetadata.uuid.in_(uuid_type_desc_df['uuid']))
        meta_col_names = [col_desc['name'] for col_desc in meta_query.column_descriptions]
        meta_df = pd.DataFrame(meta_query.all(), columns=meta_col_names)
        print("META_COLUMN_NAMES", meta_col_names)

        measurement_query = self.query(Measurement.id,
                                       Measurement.measurement_datetime,
                                       Measurement.measurement_data,
                                       Measurement.measurement_unit,
                                       Measurement.measured_method_name).filter(
                                            Measurement.id.in_(uuid_type_desc_df.index))
        measure_col_names = [col_desc['name'] for col_desc in measurement_query.column_descriptions]
        print("COLUMN_NAMES", measure_col_names)
        measurement_df = pd.DataFrame(measurement_query.all(), columns=measure_col_names)
        #measurement_df['measurement_data'] = measurement_df['measurement_data'].map(pickle.loads)
        measurement_df['measurement_data'] = measurement_df['measurement_data']

        joined_df = uuid_type_desc_df.reset_index().merge(meta_df, on='uuid')
        joined_df = joined_df.merge(measurement_df, on='id')

        return joined_df.set_index('id')

