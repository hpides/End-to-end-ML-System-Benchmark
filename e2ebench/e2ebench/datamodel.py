from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Measurement(Base):
    __tablename__ = 'measurement'

    id = Column(Integer, primary_key=True)
    uuid = Column(String, ForeignKey('benchmark_metadata.uuid'))
    measurement_datetime = Column(DateTime)
    measurement_description = Column(String, nullable=False)
    measurement_type = Column(String)
    measurement_data = Column(LargeBinary, nullable=False)
    measurement_unit = Column(String)


class BenchmarkMetadata(Base):
    __tablename__ = 'benchmark_metadata'
    uuid = Column(String, primary_key=True)
    meta_description = Column(String)
    meta_start_time = Column(DateTime)