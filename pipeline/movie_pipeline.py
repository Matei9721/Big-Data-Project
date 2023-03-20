from pyspark.sql import SparkSession
import pandas as pd

from ml.helpers import predict_lgbm, fit_bert, fit_lgbm
from openai.gpt3 import generate_plots_awards
from processing.helpers import *


def main():
    """
    Main entry points for running the compelte pipeline for loading, processing, enhancing the data followed by training
    the machine learning models and finally doing inference on the validation and test datasets.
    :return: None
    """
    # Create SparkSession
    spark = SparkSession.builder.appName('SparkCleaning').getOrCreate()

    # Load datasets in Spark
    validation_df = spark.read.option("escape", "\"").csv(
        '../datasets/gpt_data/final/validation_plots_awards_genre_no_duplicates.csv', header=True, inferSchema=True)

    test_df = spark.read.option("escape", "\"").csv(
        '../datasets/gpt_data/final/test_plots_awards_genre_no_duplicates.csv', header=True, inferSchema=True)

    wiki_df = spark.read.option("escape", "\"").csv(
        '../datasets/wiki_movie_plots_deduped.csv', header=True, inferSchema=True)

    # Add additional data and clean data
    training_df = concat_trains(spark, "../datasets/imdb/")
    wiki_df = wiki_preprocessing(wiki_df)
    training_df = preprocessing_method(training_df, wiki_df)
    validation_df = preprocessing_method(validation_df, wiki_df)
    test_df = preprocessing_method(test_df, wiki_df)

    # Generate plots using GPT3
    training_df = generate_plots_awards(training_df, "REDACTED", 128)
    validation_df = generate_plots_awards(validation_df, "REDACTED", 128)
    test_df = generate_plots_awards(test_df, "REDACTED", 128)

    # Train Bert and LGBM models
    bert_model = fit_bert(training_df, "gpu")
    lgbm_model = fit_lgbm(training_df, bert_model, "gpu")

    # Inference
    validation_prediction = predict_lgbm(validation_df, lgbm_model, bert_model, "gpu")
    test_prediction = predict_lgbm(test_df, lgbm_model, bert_model, "gpu")

    # Save results for submission
    pd.DataFrame(validation_prediction)[0].map({1: "True", 0: "False"}).to_csv(
        "../datasets/lightGBM_outputs/validation_predictions.csv", index=False, header=False)
    pd.DataFrame(test_prediction)[0].map({1: "True", 0: "False"}).to_csv(
        "../datasets/lightGBM_outputs/test_predictions.csv", index=False, header=False)


if __name__ == "__main__":
    main()
