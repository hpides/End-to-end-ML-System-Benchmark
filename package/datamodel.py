from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Measurement(Base):
    __tablename__ = 'measurement'

    id = Column(Integer, primary_key=True)
    benchmark_uuid = Column(String)
    datetime = Column(DateTime)
    function_name = Column(String)
    measurement_type = Column(String)
    value = Column(String)

