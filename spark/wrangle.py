import pandas as pd
import numpy as np
from pydataset import data
import pyspark

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import concat, col, sum, avg, min, max, mean, count
from pyspark.sql.functions import to_date
from pyspark.sql.functions import *


def wrangle_311():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    #SA 311 data sources
    df = spark.read.csv('case.csv',header=True,inferSchema=True)
    source = spark.read.csv('source.csv',inferSchema=True,header=True)
    dept = spark.read.csv('dept.csv',inferSchema=True,header=True)
    #changing join names on dept and source
    dept = dept.selectExpr('dept_division as dept_div','dept_name')
    source = source.selectExpr('source_id as id','source_username')
    
    #changing date time
    df = df.select('dept_division','case_id','case_late','num_days_late','service_request_type','SLA_days','case_status','source_id','request_address','council_district',to_date(df.case_opened_date,'M/d/yy').alias('case_open_date')\
             ,to_date(df.case_closed_date,'M/d/yy').alias('case_close_date')\
             ,to_date(df.SLA_due_date,'M/d/yy').alias('case_due_date'))
    df = df.select('*',col('council_district').cast('string'))
    df = df.select('*',year('case_close_date').alias('year_closed'))

    #joining dept and source tables to 311_case csv
    df = df.join(dept,df.dept_division == dept.dept_div,how='left')
    df = df.join(source,df.source_id == source.id,how='left')
    return df