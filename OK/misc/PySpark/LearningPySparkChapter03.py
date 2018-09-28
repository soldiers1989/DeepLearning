import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

sc = SparkContext(appName="LearningPySparkChapter03")
spark = SparkSession.builder.master("local").appName("Python Spark SQL basic example").config("spark.sql.warehouse.dir","file:///").getOrCreate()

#stringJSONRDD = sc.parallelize((""" 
#  { "id": "123",
#    "name": "Katie",
#    "age": 19,
#    "eyeColor": "brown"
#  }""",
#   """{
#    "id": "234",
#    "name": "Michael",
#    "age": 22,
#    "eyeColor": "green"
#  }""",
#  """{
#    "id": "345",
#    "name": "Simone",
#    "age": 23,
#    "eyeColor": "blue"
#  }""")
#)
#print(stringJSONRDD.take(1))
#
#swimmersJSON = spark.read.json(stringJSONRDD)
#swimmersJSON.createOrReplaceTempView("swimmersJSON")
#swimmersJSON.show()


## Generate our own CSV data 
##  This way we don't have to access the file system yet.
#stringCSVRDD = sc.parallelize([(123, 'Katie', 19, 'brown'), (234, 'Michael', 22, 'green'), (345, 'Simone', 23, 'blue')])
#
## The schema is encoded in a string, using StructType we define the schema using various pyspark.sql.types
#schemaString = "id name age eyeColor"
#schema = StructType([
#    StructField("id", LongType(), True),    
#    StructField("name", StringType(), True),
#    StructField("age", LongType(), True),
#    StructField("eyeColor", StringType(), True)
#])
#
## Apply the schema to the RDD and Create DataFrame
#swimmers = spark.createDataFrame(stringCSVRDD, schema)
#
## Creates a temporary view using the DataFrame
#swimmers.createOrReplaceTempView("swimmers")
#
#swimmers.printSchema()
#
#print('*'*10+"select * from swimmers")
#spark.sql("select * from swimmers").show()
#print('*'*10+"select count(1) from swimmers")
#spark.sql("select count(1) from swimmers").show()
#print('*'*10+"select filter")
#swimmers.select("id", "age").filter("age = 22").show()
#print('*'*10+"select id, age from swimmers where age = 22")
#spark.sql("select id, age from swimmers where age = 22").show()
#print('*'*10+"select name, eyeColor from swimmers where eyeColor like 'b%'")
#spark.sql("select name, eyeColor from swimmers where eyeColor like 'b%'").show()

flightPerfFilePath = "data/departuredelays.csv"
airportsFilePath = "data/airport-codes-na.txt"

# Obtain Airports dataset
airports = spark.read.csv(airportsFilePath, header='true', inferSchema='true', sep='\t')
airports.createOrReplaceTempView("airports")

# Obtain Departure Delays dataset
flightPerf = spark.read.csv(flightPerfFilePath, header='true')
flightPerf.createOrReplaceTempView("FlightPerformance")

# Cache the Departure Delays dataset 
flightPerf.cache()

print('*'*10+"select a.City, f.origin.... join")
spark.sql("select a.City, f.origin, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.State = 'WA' group by a.City, f.origin order by sum(f.delay) desc").show()

print('*'*10+"select a.City, sum(f.delay)....join....group by")
spark.sql("select a.State, sum(f.delay) as Delays from FlightPerformance f join airports a on a.IATA = f.origin where a.Country = 'USA' group by a.State ").show()

sc.stop()

