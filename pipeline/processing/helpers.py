import pyspark.sql
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, udf, when, split, regexp_replace, min, mean, lower, array_contains
from pyspark.ml.feature import VectorAssembler, StandardScaler

import os
from unidecode import unidecode


def scaling_method(df, column_name, vector_column_name, scaled_column_name, placeholder):
    """
    Scales the data to improve ML performance
    :param df: Pyspark Dataframe
    :param column_name: Name of the column
    :param vector_column_name: Same as above but vectorised
    :param scaled_column_name: Same as above but scaled
    :param placeholder: Placeholder value
    :return:
    """
    # Replace null values with a placeholder value, e.g. -1, using when/otherwise
    df = df.withColumn(column_name, when(col(column_name).isNull(), placeholder).otherwise(col(column_name)))

    # Create a VectorAssembler to convert the scalar column to a vector column
    assembler = VectorAssembler(inputCols=[column_name], outputCol=vector_column_name)
    df = assembler.transform(df)

    # Create the StandardScaler transformer and fit it to the data
    scaler = StandardScaler(inputCol=vector_column_name, outputCol=scaled_column_name, withMean=True, withStd=True)
    scaler_model = scaler.fit(df)

    scaled_data = scaler_model.transform(df)

    # Replace the placeholder values with null
    scaled_data = scaled_data.withColumn(scaled_column_name, when(col(column_name) == -1, None).
                                         otherwise(col(scaled_column_name)))
    scaled_data = scaled_data.withColumn(column_name, when(col(column_name) == -1, None).
                                         otherwise(col(column_name)))

    scaled_data = scaled_data.drop(vector_column_name)

    return scaled_data


def wiki_preprocessing(wiki) -> pyspark.sql.DataFrame:
    """
    Method that applies pre-processing Pyspark functions on the wiki dataframe to allow merging
    :param wiki: Wikipedia movie dataframe
    :return: pyspark.sql.DataFrame
    """
    udf_unidecode = udf(lambda x: unidecode(x) if x is not None else None)

    wiki = wiki.withColumn("Title", lower(col("Title")))
    wiki = wiki.withColumnRenamed("Title", "primaryTitle")
    wiki = wiki.withColumnRenamed("Release Year", "startYear")
    wiki = wiki.withColumn("primaryTitle", udf_unidecode(wiki["primaryTitle"]))
    wiki = wiki.dropDuplicates(subset=["primaryTitle"])

    return wiki


def genre_preprocessing(df) -> pyspark.sql.DataFrame:
    """

    :param df: Initial dataframe
    :return: pyspark.sql.DataFrame
    """
    # Everything to lower case
    df = df.withColumn('Genre', lower(col('Genre')))

    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'film ', ''))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), ' in ', ' '))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), ' of ', ' '))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), '\.', ''))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'drama based on the novel by russell banks;', 'drama'))
    df = df.withColumn('Genre',
                       regexp_replace(col('Genre'), 'drama adapted from wajdi mouawad\'s play of the same name',
                                      'drama'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'world war ii', 'war'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'biopic', 'biography'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'drama\[not in citation given\]', 'drama'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'kung fu', 'kungfu'))
    df = df.withColumn('Genre',
                       regexp_replace(col('Genre'), 'historical romance based on colm tóibín\'s novel of the same name',
                                      'romantic'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), '3-d', '3d'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'neo-noir', 'noir'))

    df = df.withColumn('Genre', when(col('Genre').isNull(), 'unknown').otherwise(col('Genre')))

    df = df.withColumn('Genre', regexp_replace(col('Genre'), '/', ', '))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), ' ,', ','))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), ', ', ','))

    df = df.withColumn('Genre', regexp_replace(col('Genre'), '-', ' '))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'sci fi', 'sci-fi'))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), ' ', ','))
    df = df.withColumn('Genre', regexp_replace(col('Genre'), 'martial,arts', 'martial arts'))

    # Split the columns
    df = df.withColumn('Genre', split(lower(col('Genre')), ','))

    print(df.columns)

    # Creating the one hot encoding
    unique_values = [str(row[0]) for row in df.selectExpr("explode(array_distinct(Genre))").distinct().collect()]
    for value in unique_values:
        df = df.withColumn(value, array_contains('Genre', value).cast('int'))

    df = df.drop('&', ' ', 'on', 'sf', 'i', 'jidaigeki', '', 'the', 'mouawad\'s', 'tóibín\'s', 'drama[not', 'given]',
                 'wajdi')

    return df


def preprocessing_method(df, wiki_df) -> pyspark.sql.DataFrame:
    """
    This method will convert to integer all the years and will put to categorical weather
    a movie won any awards or not.
    param df: dataframe
    return: pyspark.sql.DataFrame
    """


    # Bringing to lower case the titles
    scalar_udf = udf(lambda arr: float(arr[0]), DoubleType())
    udf_unidecode = udf(lambda x: unidecode(x) if x is not None else None)

    df = df.withColumn("startYear", regexp_replace(col("startYear"), r"\\N", str(None)))
    df = df.fillna({'startYear': 'endYear'}, subset=['startYear'])
    df = df.withColumn("startYear", col("startYear").cast("int"))
    df = df.withColumn("primaryTitle", udf_unidecode(df["primaryTitle"]))

    df = df.withColumn('primaryTitle', lower(col('primaryTitle')))
    df = df.withColumn('originalTitle', lower(col('originalTitle')))

    # Here casting to integer all the numerical values
    df = df.withColumn('startYear', col('startYear').cast('int'))
    df = df.withColumn('endYear', col('endYear').cast('int'))
    df = df.withColumn('numVotes', col('numVotes').cast('int'))
    df = df.withColumn('runtimeMinutes', col('runtimeMinutes').cast('int'))

    # Preprocessing years. We add range of years and then for each row we check in which range it is
    ranges = ['1915-1929', '1930-1939', '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989',
              '1990-1999', '2000-2009', '2010-2019', '2020-2023']

    for r in ranges:
        limit0, limit1 = r.split('-')
        limit0 = int(limit0)
        limit1 = int(limit1)
        df = df.withColumn(r, when((col('startYear') >= limit0) & (col('startYear') <= limit1), 1).otherwise(0))

    # Preprocess runtime minutes. If there are null values substitute with the mean.
    mean_runtime = df.select(mean("runtimeMinutes").cast("int")).collect()[0][0]
    df = df.withColumn("runtimeMinutes", when(col("runtimeMinutes").isNull(), mean_runtime).
                       otherwise(col("runtimeMinutes")))

    # Preprocessing of the numVotes.
    min_numVotes = df.select(min('numVotes')).collect()[0][0]
    df = df.withColumn('numVotes', when(col('numVotes').isNull(), min_numVotes).
                       otherwise(col('numVotes')))

    df = scaling_method(df, "numVotes", "vector_column_votes", "scaled_votes", -1)
    df = df.withColumn('numVotes', scalar_udf(df['scaled_votes']))

    df = df.join(wiki_df.select('primaryTitle', 'Genre', 'startYear'), on=['primaryTitle', 'startYear'], how='left')

    # Preprocessing of genre. Create also a list of genres.
    df = genre_preprocessing(df)

    # Dropping the columns.
    df = df.drop('originalTitle', 'endYear', 'scaled_votes', 'Genre')
    print(df.columns)

    return df


def concat_trains(spark, dir_path) -> pyspark.sql.DataFrame:
    """
    This method concatenated all training files in the specified directory
    :param dir_path: Path to the files
    :param spark:
    :return: pyspark.sql.DataFrame
    """
    # Get a list of all the files in the directory
    files = os.listdir(dir_path)

    # Filter the list to include only files that start with "train-" and have a ".csv" extension
    train_files = [f for f in files if f.startswith("train-") and f.endswith(".csv")]

    # Count the number of files
    num_train_files = len(train_files)

    print("Number of train files: {}".format(num_train_files))

    # Create an empty list to store the DataFrames
    dfs = []

    # Loop through the file names and read each file into a DataFrame
    for file in train_files:
        file_name = f"../datasets/imdb/{file}"
        df = spark.read.option("escape", "\"").csv(file_name, header=True, inferSchema=True)
        dfs.append(df)

    # Loop through the DataFrames and print the shape of each one
    for i, df in enumerate(dfs):
        num_rows = df.count()
        num_cols = len(df.columns)
        print("Shape of df_train_{}: {} rows, {} columns".format(i + 1, num_rows, num_cols))

    # Union all the DataFrames into one big DataFrame
    big_df = dfs[0]
    for df in dfs[1:]:
        big_df = big_df.union(df)

    # Drop columns _c0
    big_df = big_df.drop("_c0")

    big_df.cache()
    num_rows = big_df.count()
    num_cols = len(big_df.columns)
    print(f"\nShape of big_df: {num_rows} rows, {num_cols} columns")

    return big_df
