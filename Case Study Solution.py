# Databricks notebook source
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import requests

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# COMMAND ----------

df = spark.read.json("s3a://csparkdata/ol_cdump.json")

# COMMAND ----------

df = df.withColumn("publish_date", f.regexp_replace(f.col("publish_date"), "([0-9])th", "$1"))

# COMMAND ----------

df = df.withColumn("publish_date_formatted", f.expr('''
case
    when lower(publish_date) like "%u" then null
    when to_date(publish_date, "dd MMM yyyy") is not null and to_date(publish_date, "dd MMM yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "dd MMM yyyy")
    when to_date(publish_date, "dd/MM/yyyy") is not null and to_date(publish_date, "dd/MM/yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "dd/MM/yyyy")
    when to_date(publish_date, "MM/dd/yyyy") is not null and to_date(publish_date, "MM/dd/yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MM/dd/yyyy")
    when to_date(publish_date, "MM-dd-yyyy") is not null and to_date(publish_date, "MM-dd-yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MM-dd-yyyy")
    when to_date(publish_date, "MM.dd.yyyy") is not null and to_date(publish_date, "MM.dd.yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MM.dd.yyyy")
    when to_date(publish_date, "MM/dd/yy") is not null and to_date(publish_date, "MM/dd/yy") <= to_date(current_timestamp()) then to_date(publish_date, "MM/dd/yy")
    when to_date(publish_date, "MM-dd-yy") is not null and to_date(publish_date, "MM-dd-yy") <= to_date(current_timestamp()) then to_date(publish_date, "MM-dd-yy")
    when to_date(publish_date, "MM/yyyy") is not null and to_date(publish_date, "MM/yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MM/yyyy")
    when to_date(publish_date, "MMMM d, yyyy") is not null and to_date(publish_date, "MMMM d, yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MMMM d, yyyy")
    when to_date(publish_date, "MMMM d,yy") is not null and to_date(publish_date, "MMMM d,yy") <= to_date(current_timestamp()) then to_date(publish_date, "MMMM d,yy")
    when to_date(publish_date, "MMMM yyyy") is not null and to_date(publish_date, "MMMM yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MMMM yyyy")
    when to_date(publish_date, "MMMM, yyyy") is not null and to_date(publish_date, "MMMM, yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "MMMM, yyyy")
    when to_date(publish_date, "yyyy") is not null and to_date(publish_date, "yyyy") <= to_date(current_timestamp()) then to_date(publish_date, "yyyy")
    else null
end
'''))

# COMMAND ----------

df.createOrReplaceTempView("df")

# COMMAND ----------

df_w_pub_year = spark.sql('''
  select *,
  case 
    when publish_date_formatted is not null then year(publish_date_formatted) 
    else 0000
  end as publish_year
  from df
''')

# COMMAND ----------

df_w_pub_year.createOrReplaceTempView("df_pub_year")

# COMMAND ----------

df_cleaned = spark.sql('''
  select * from df_pub_year 
  where title is not null
  and lower(title) not like "%no title exists%"
  and number_of_pages > 20
  and publish_year > 1950
  and publish_date_formatted is not null
  and key is not null
  and authors is not null
''')

# COMMAND ----------

def parse_author_keys(author_keys):
  return ",".join([x[1].split("/")[2] for x in author_keys])

parse_author_keys_udf = f.udf(parse_author_keys)
spark.udf.register("parse_author_keys_udf", parse_author_keys)

# COMMAND ----------

def parse_book_key(book_key):
  return book_key.split("/")[2]

parse_book_key_udf = f.udf(parse_book_key)
spark.udf.register("parse_book_key_udf", parse_book_key)

# COMMAND ----------

def get_author_name(author_key):
  url = f"https://openlibrary.org/authors/{author_key}.json"
  try:
    r = requests.get(url = url)
    author_name = r.json()["name"]
    return author_name
  except:
    return None
  
get_author_name_udf = f.udf(get_author_name)
spark.udf.register("get_author_name_udf", get_author_name)

# COMMAND ----------

df_cleaned = df_cleaned.withColumn("author_keys", parse_author_keys_udf(f.col("authors")))

# COMMAND ----------

df_cleaned = df_cleaned.withColumn("book_key", parse_book_key_udf(f.col("key")))

# COMMAND ----------

df_cleaned.createOrReplaceTempView("data")

# COMMAND ----------

ques1 = spark.sql('''
  select distinct book_key, title as book_title, number_of_pages from
  (select *, dense_rank() over (order by number_of_pages desc) as page_rank from data)
  where page_rank = 1
''').toPandas()

# COMMAND ----------

ques1

# COMMAND ----------

ques2_inter = spark.sql('''
  select *, explode(genres) as genre from data
''')

ques2_inter = ques2_inter.withColumn("genre", f.regexp_replace(f.col("genre"), "\.$", ""))

ques2_inter.createOrReplaceTempView("ques2_inter")

ques2 = spark.sql('''
  select * from (
  select rank() over (order by books desc) as genre_rank, genre, books from
  (select genre, count(distinct key) as books from ques2_inter group by 1)
  )
  order by genre_rank
''').toPandas()

# COMMAND ----------

ques2[ques2.genre_rank <= 5]

# COMMAND ----------

ques3_inter = spark.sql('''
  select *, explode(split(author_keys, ",")) as author_key from data
''')

ques3_inter.createOrReplaceTempView("ques3_inter")

ques3 = spark.sql('''
  select author_rank, author_key, author_name_data, get_author_name_udf(author_key) as author_name_site, books from (
  select *, rank() over (order by books desc) as author_rank from
  (select author_key, name as author_name_data, count(distinct key) as books from ques3_inter group by 1,2)
  )
  where author_rank <= 5
  order by author_rank
''').toPandas()

# COMMAND ----------

ques3

# COMMAND ----------

ques4 = spark.sql('''
  select publish_year, count(distinct author_key) as authors from
  (select *, explode(split(author_keys, ",")) as author_key from data)
  group by 1
  order by 1
''').toPandas()

# COMMAND ----------

pd.set_option("display.max_rows", None)
ques4

# COMMAND ----------

ques5 = spark.sql('''
  select date_format(publish_date_formatted, "yyyy-MM") as publish_year_month, count(distinct author_key) as authors, count(distinct key) as books from
  (select *, explode(split(author_keys, ",")) as author_key from data)
  group by 1
  order by 1
''').toPandas()

# COMMAND ----------

ques5
