import pyspark.sql
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, udf, when, split, regexp_replace, min, mean, lower, array_contains
from pyspark.ml.feature import VectorAssembler, StandardScaler

import os
from unidecode import unidecode

from pyspark.ml import Transformer
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import col, when, lower, mean, udf, split, regexp_replace, \
    min, array_contains
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler

ranges = ('1915-1929', '1930-1939', '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989',
          '1990-1999', '2000-2009', '2010-2019', '2020-2023')
# %%
scalar_udf = udf(lambda arr: float(arr[0]), DoubleType())


class FillNATransformer(Transformer):
    def __init__(self, input_cols=None, output_cols=None):
        super(FillNATransformer, self).__init__()
        self.input_cols = input_cols
        self.output_cols = output_cols

    def _transform(self, df):
        df = df.withColumn(self.output_cols[0], regexp_replace(self.input_cols[0], r"\\N", str(None)))
        df = df.fillna({self.output_cols[0]: self.output_cols[1]}, subset=[self.input_cols[0]])
        df = df.withColumn(self.output_cols[0], self.input_cols[0].cast("int"))
        return df


class LowerCaseNormaliseTransformer(Transformer):
    def __init__(self, input_cols=None, output_cols=None):
        super(LowerCaseNormaliseTransformer, self).__init__()
        self.input_cols = input_cols
        self.output_cols = output_cols

    def _transform(self, df):
        udf_unidecode = udf(lambda x: unidecode(x) if x is not None else None)
        for index in enumerate(self.output_cols):
            df = df.withColumn(self.output_cols[index], udf_unidecode(df[self.output_cols[index]]))
            df = df.withColumn(self.output_cols[index], lower(self.input_cols[index]))
            df = df.withColumn(self.output_cols[index], lower(self.input_cols[index]))

        return df


class CastToInt(Transformer):
    def __init__(self, input_cols=None, output_cols=None):
        super(CastToInt, self).__init__()
        self.input_cols = input_cols
        self.output_cols = output_cols

    def _transform(self, df):
        df = df.withColumn(self.output_cols[0], col(self.input_cols[0]).cast('int'))
        df = df.withColumn(self.output_cols[1], col(self.input_cols[1]).cast('int'))
        df = df.withColumn(self.output_cols[2], col(self.input_cols[2]).cast('int'))
        df = df.withColumn(self.output_cols[3], col(self.input_cols[3]).cast('int'))

        return df


class PreprocessYears(Transformer):
    def __init__(self, input_col=None, output_col=None, ranges=None):
        super(PreprocessYears, self).__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.ranges = ranges

    def _transform(self, df):
        for r in self.ranges:
            limit0, limit1 = r.split('-')
            limit0 = int(limit0)
            limit1 = int(limit1)
            df = df.withColumn(r, when((col('startYear') >= limit0) & (col('startYear') <= limit1), 1).otherwise(0))

        return df


class PrepRuntimeMin(Transformer):
    def __init__(self, input_col=None, output_col=None):
        super(PrepRuntimeMin, self).__init__()
        self.input_col = input_col
        self.output_col = output_col

    def _transform(self, df):
        mean_runtime = df.select(mean(self.input_col).cast("int")).collect()[0][0]
        df = df.withColumn(self.input_col, when(col(self.input_col).isNull(), mean_runtime).
                           otherwise(col(self.input_col)))

        return df


class PrepNumVotes(Transformer):
    def __init__(self, input_col=None, output_col=None):
        super(PrepNumVotes, self).__init__()
        self.input_col = input_col
        self.output_col = output_col

    def __scaling_method(self, df, column_name, vector_column_name, scaled_column_name, placeholder):
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

    def _transform(self, df):
        min_votes = df.select(min(self.input_col)).collect()[0][0]
        df = df.withColumn(self.input_col, when(col(self.input_col).isNull(), min_votes).
                           otherwise(col(self.input_col)))
        df = self.__scaling_method(df, self.input_col, "vector_column_votes", "scaled_votes", -1)
        df = df.withColumn(self.input_col, scalar_udf(df['scaled_votes']))

        return df


class PrepGenre(Transformer):
    def __init__(self):
        super(PrepGenre, self).__init__()

    def __genre_preprocess(self, df):
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
        df = df.withColumn('Genre', regexp_replace(col('Genre'),
                                                   'historical romance based on colm tóibín\'s novel of the same name',
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

        # Creating the one hot encoding
        unique_values = [str(row[0]) for row in df.selectExpr("explode(array_distinct(Genre))").distinct().collect()]
        for value in unique_values:
            df = df.withColumn(value, array_contains('Genre', value).cast('int'))

        df = df.drop('&', ' ', 'on', 'sf', 'i', 'jidaigeki', '', 'the', 'mouawad\'s', 'tóibín\'s', 'drama[not',
                     'given]', 'wajdi')

        return df

    def _transform(self, df):
        df = self.__genre_preprocess(df)
        return df


class WikiProcessor(Transformer):
    def __init__(self, input_cols=None, output_cols=None):
        super(WikiProcessor, self).__init__()
        self.input_cols = input_cols
        self.output_cols = output_cols

    def _transform(self, df):
        udf_unidecode = udf(lambda x: unidecode(x) if x is not None else None)

        df = df.withColumn("Title", lower(col("Title")))
        df = df.withColumnRenamed("Title", "primaryTitle")
        df = df.withColumnRenamed("Release Year", "startYear")
        df = df.withColumn("primaryTitle", udf_unidecode(df["primaryTitle"]))
        df = df.dropDuplicates(subset=["primaryTitle"])
        return df


def preprocessing_method(df, wiki_df) -> pyspark.sql.DataFrame:
    """
    This method will convert to integer all the years and will put to categorical weather
    a movie won any awards or not.
    param df: dataframe
    return: pyspark.sql.DataFrame
    """
    # Create PySpark transformers
    fill_NA = FillNATransformer(input_cols=["startYear", "endYear"],
                                output_cols=["startYear", "endYear"])
    lowercase = LowerCaseNormaliseTransformer(input_cols=["primaryTitle", "originalTitle"],
                                              output_cols=["primaryTitle", "originalTitle"])
    integer = CastToInt(input_cols=["startYear", "endYear", "numVotes", "runtimeMinutes"],
                        output_cols=["startYear", "endYear", "numVotes", "runtimeMinutes"])
    year = PreprocessYears(input_col="startYear", output_col="startYear", ranges=ranges)
    runtime = PrepRuntimeMin(input_col="runtimeMinutes", output_col="runtimeMinutes")
    votes = PrepNumVotes(input_col="numVotes", output_col="numVotes")
    genre = PrepGenre()

    # Create pipeline and run transformers to clean data
    pipeline = Pipeline(stages=[fill_NA, lowercase, integer, year, runtime, votes])
    model = pipeline.fit(df)
    df_prep = model.transform(df)

    # Now we can join with wikipedia dataset
    df_prep = df_prep.join(wiki_df.select('primaryTitle', 'Genre', 'startYear'), on=['primaryTitle', 'startYear'],
                           how='left')

    # Run transformer on genres
    df_prep = genre.transform(df_prep)

    # Drop unneeded data
    df_prep = df_prep.drop('originalTitle', 'endYear', 'scaled_votes', 'Genre')

    return df_prep


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
