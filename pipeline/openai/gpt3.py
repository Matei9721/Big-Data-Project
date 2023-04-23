import pyspark.sql

from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import udf, lit

import openai


def generate_plots_awards(spark_df, openai_key, max_tokens) -> pyspark.sql.DataFrame:
    """
    Function that uses the OpenAI API to generate plots for the movies using GPT3
    :param spark_df: Spark dataframe
    :param openai_key: OpenAI API key
    :param max_tokens: How long the plots should be
    :return: pyspark.sql.DataFrame
    """
    openai.api_key = openai_key
    spark_df = spark_df.withColumn("plot", lit(""))

    def generate_plot(primaryTitle, startYear, runtimeMinutes):

        prompt = f"Give me a roughly 100 words plot of the {primaryTitle} from {startYear} that has a duration" \
                 f" of {runtimeMinutes} minutes"
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0
            )

            return response["choices"][0].text

        except Exception as e:
            print(f"Failed on one query for movie {primaryTitle}. Continuing., ", e)

    def define_awards(primaryTitle, originalTitle, startYear, runtimeMinutes):

        prompt = f"Did the movie {primaryTitle if primaryTitle is not None else originalTitle} from {startYear} " \
                 f"with a runtime of {runtimeMinutes} win any awards? Please only answer with Yes if it won any" \
                 f" awards or with No if it did not."

        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=32,
                top_p=1,
                frequency_penalty=0.5,
                presence_penalty=0
            )

            return 1 if "Yes" in response["choices"][0] else 0

        except Exception as e:
            print("Failed on one query. Continuing., ", e)

    # Generation of the udf
    udf_generate_plot = udf(generate_plot, StringType())
    udf_awards = udf(define_awards, IntegerType())

    spark_df = spark_df.withColumn('awards', udf_awards(spark_df["primaryTitle"], spark_df["originalTitle"],
                                                        spark_df["startYear"], spark_df["runtimeMinutes"]))

    spark_df = spark_df.withColumn('plot', udf_generate_plot(spark_df["primaryTitle"], spark_df["startYear"],
                                                             spark_df["runtimeMinutes"]))

    return spark_df
