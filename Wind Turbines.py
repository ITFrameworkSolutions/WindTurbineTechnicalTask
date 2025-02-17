# Databricks notebook source
# MAGIC %md
# MAGIC # Brief
# MAGIC Consider the following scenario:
# MAGIC
# MAGIC You are a data engineer for a renewable energy company that operates a farm of wind turbines. The turbines generate power based on wind speed and direction, and their output is measured in megawatts (MW). Your task is to build a data processing pipeline that ingests raw data from the turbines and performs the following operations:
# MAGIC
# MAGIC - Cleans the data: The raw data contains missing values and outliers, which must be removed or imputed.
# MAGIC - Calculates summary statistics: For each turbine, calculate the minimum, maximum, and average power output over a given time period (e.g., 24 hours).
# MAGIC - Identifies anomalies: Identify any turbines that have significantly deviated from their expected power output over the same time period. Anomalies can be defined as turbines whose output is outside of 2 standard deviations from the mean.
# MAGIC - Stores the processed data: Store the cleaned data and summary statistics in a database for further analysis.
# MAGIC
# MAGIC Data is provided to you as CSVs which are appended daily. Due to the way the turbine measurements are set up, each csv contains data for a group of 5 turbines. Data for a particular turbine will always be in the same file (e.g. turbine 1 will always be in data_group_1.csv). Each day the csv will be updated with data from the last 24 hours, however the system is known to sometimes miss entries due to sensor malfunctions.
# MAGIC
# MAGIC The files provided in the attachment represent a valid set for a month of data recorded from the 15 turbines. Feel free to add/remove data from the set provided in order to test/satisfy the requirements above.
# MAGIC
# MAGIC Your pipeline should be scalable and testable; emphasis is based on the clarity and quality of the code and the implementation of the functionality outlined above, and not on the overall design of the application.
# MAGIC
# MAGIC Your solution should be implemented in Python, using any frameworks or libraries that you deem appropriate. Please provide a brief description of your solution design and any assumptions made in your implementation.
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Outline
# MAGIC
# MAGIC ### Considerations
# MAGIC Data is stored on an AWS S3 bucket 
# MAGIC
# MAGIC Initial load wont have enough data to apply an average. Flag rows and revisit.
# MAGIC
# MAGIC Malfunctions occur. This would be represented as missing or corrupted data. Options
# MAGIC - Exclude and replace with average - chosen option
# MAGIC - Replace with data that arrived late
# MAGIC - Exclude entire day
# MAGIC
# MAGIC ### Assumptions
# MAGIC All day measurements use timestamps that assume 0:00:00 to 23:59:59. If the feeds are across timezones, this would need to be adapted
# MAGIC
# MAGIC ### Future
# MAGIC Use wind_direction, wind_speed, power_output and malfuntions to attempt to inform or predict wear and tear or potential future malfunctions.
# MAGIC
# MAGIC Consider archiving the feeds to reduce processing as the data scales.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Code

# COMMAND ----------

# DBTITLE 1,Ingest Data
from pyspark.sql.functions import col, regexp_replace, split
# specify AWS S3 bucket 
bucket_url = "s3a://justanotherrandoms3bucket/test_data/"

# read csvs into dataframe 
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(bucket_url + "*.csv") 

display(df.count())
display(df)


# COMMAND ----------

# DBTITLE 1,Identify Anomalies
from pyspark.sql.functions import mean, stddev, to_date, col, hour, round

# Define outliers columns
outliers = ['power_output']

# Calculate mean and stddev grouped by turbine_id
stats = df.groupBy("turbine_id").agg(
    *[mean(c).alias(c + '_mean') for c in outliers] + 
    [stddev(c).alias(c + '_stddev') for c in outliers]
)

# Join the stats back to the original dataframe
df_with_stats = df.join(stats, on="turbine_id") 

display(stats)
display(df_with_stats)



# COMMAND ----------

# DBTITLE 1,Clean Data

# Remove outliers (values outside 2 standard deviations)
for col_name in outliers:
    mean_col = col(col_name + '_mean')
    stddev_col = col(col_name + '_stddev')
    cleaned_df = df_with_stats.filter(
        (col(col_name) >= mean_col - 2 * stddev_col) & 
        (col(col_name) <= mean_col + 2 * stddev_col)
    ).drop(col_name + '_mean', col_name + '_stddev')
    
# store removed outliers
removed_outliers_df = df.join(cleaned_df.select("turbine_id", "timestamp").distinct(), ["turbine_id", "timestamp"], "left_anti")

display(cleaned_df.count())
display(cleaned_df)
display(removed_outliers_df)

# COMMAND ----------

# DBTITLE 1,Check for Missing Hours in a Day for Each Turbine
from pyspark.sql.functions import min, max, sequence, explode, col, lit, expr, coalesce

# Get the range of timestamps in the data
min_timestamp = df.select(min("timestamp")).first()[0]
max_timestamp = df.select(max("timestamp")).first()[0]

# Generate a list of all hours within the range for each turbine
turbine_ids = cleaned_df.select("turbine_id").distinct().collect()
hour_range_df = spark.createDataFrame([(min_timestamp,)], ["start_timestamp"]) \
    .selectExpr("explode(sequence(start_timestamp, timestamp'{}', interval 1 hour)) as timestamp".format(max_timestamp))

# Cross join with turbine IDs to get all possible hour-turbine combinations
turbine_hour_range_df = hour_range_df.crossJoin(spark.createDataFrame(turbine_ids))

# Find missing hours for each turbine
missing_hours_df = turbine_hour_range_df.join(cleaned_df.select("timestamp","turbine_id").distinct(), ["turbine_id", "timestamp"], "left_anti")

# Display missing hours
display(missing_hours_df)


# COMMAND ----------

# DBTITLE 1,Append missing hours
# Append missing hours with average values to cleaned_df
cleaned_with_missing_hours_df = cleaned_df.unionByName(missing_hours_df, allowMissingColumns=True) \
                                          .join(stats.select("turbine_id", expr("round(power_output_mean, 1) as power_output_mean")), 
                                                on=["turbine_id"], 
                                                how="left") \
                                          .withColumn("power_output", coalesce(col("power_output"), col("power_output_mean"))) \
                                          .drop('power_output_mean') \
                                          .withColumn("date", to_date(col("timestamp")))

# Display updated cleaned_df
display(cleaned_with_missing_hours_df)

# COMMAND ----------

# DBTITLE 1,Calculate Summary Statistics
from pyspark.sql.functions import min, max, avg
# Calculate Summary Statistics
summary_df = cleaned_with_missing_hours_df.groupBy("turbine_id","date").agg(
    min("power_output").alias("min_power"),
    max("power_output").alias("max_power"),
    avg("power_output").alias("avg_power")
).orderBy("date","turbine_id")

display(summary_df.orderBy("turbine_id","date"))


# COMMAND ----------

# DBTITLE 1,Store data to tables
# Save the cleaned_with_missing_hours_df DataFrame to a table
cleaned_with_missing_hours_df.write.mode("overwrite").saveAsTable("cleaned_turbine_data")

# Save the summary_df DataFrame to a table
summary_df.write.mode("overwrite").saveAsTable("turbine_summary_statistics")