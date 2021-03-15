from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Measurement(Base):
    __tablename__ = 'measurement'

    id = Column(Integer, primary_key=True)
    benchmark_uuid = Column(String, ForeignKey('benchmark_metadata.uuid'))
    datetime = Column(DateTime)
    description = Column(String, nullable=False)
    measurement_type = Column(String)
    value = Column(String, nullable=False)
    unit = Column(String)


class BenchmarkMetadata(Base):
    __tablename__ = 'benchmark_metadata'
    uuid = Column(String, primary_key=True)
    description = Column(String)
    start_time = Column(DateTime)