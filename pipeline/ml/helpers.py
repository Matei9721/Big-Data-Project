import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from pyspark.sql.functions import col
import numpy as np
import lightgbm as lgb


def fit_bert(train_df, device):
    """
    Fits the HuggingFace BERT model on train data
    :param train_df: Training Spark dataframe
    :param device: Cpu or Gpu
    :return: BERT model
    """
    train = train_df.toPandas()
    train = train[['plot', 'label']]

    max_len = 128  # maximum sequence length to use

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded_plots = tokenizer.batch_encode_plus(list(train['plot'].values), add_special_tokens=True, max_length=max_len,
                                                padding=True, truncation=True, return_tensors='pt')
    labels_tensor = torch.tensor(train['label'])

    # Load the pre-trained BERT model and add a classification layer on top of it
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Hyperparams
    batch_size = 16
    epochs = 3

    dataset = TensorDataset(encoded_plots['input_ids'], encoded_plots['attention_mask'], labels_tensor)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Fine-tune the BERT model
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()  # set the gradients to zero for each batch

            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # still don't know what it does

            # parameter update based on the current gradient (stored in . grad attribute of a parameter)
            optimizer.step()

            # update learning rate with scheduler
            scheduler.step()

            if i % 50 == 0:
                print(f'epoch:{epoch}, batch {i} out of {len(dataset) // batch_size} is done')

    return model


def predict_bert(data, model, device):
    """
    Uses the BERT model to generate the probabilities of a movie
    :param data: Spark dataframe
    :param model: BERT trained model
    :param device: Cpu or Gpu
    :return: list
    """
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_unseen_data = tokenizer.batch_encode_plus(data,
                                                      add_special_tokens=True,
                                                      max_length=128,
                                                      padding=True,
                                                      truncation=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')

    unseen_dataset = TensorDataset(encoded_unseen_data['input_ids'], encoded_unseen_data['attention_mask'])
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=16)

    model.eval()
    predictions = []
    bert_variable = []
    with torch.no_grad():
        for batch in unseen_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, batch_predictions = torch.max(logits, dim=1)
            predictions.extend(batch_predictions.tolist())

            probabilities = F.softmax(logits, dim=1)[:, 1]  # bertvariable
            bert_variable.extend(probabilities.tolist())

    return bert_variable


def fit_lgbm(train_df):
    """
    Fits the LGBM model on train data
    :param train_df: Spark dataframe
    :return: LGBM model
    """
    train = train_df.toPandas()
    genres_to_keep = ['drama', 'horror', 'biography', 'noir', 'comedy', 'unknown', 'sci-fi']
    all_genres = train.iloc[:, 18:].columns.values
    genres_to_remove = set(all_genres) - set(genres_to_keep)
    train = train.drop(list(genres_to_remove), axis=1)

    y_train = train["label"]
    X_train = train.drop(["label"], axis=1)

    # Create lgbm object
    clf = lgb.LGBMClassifier()

    # Train lgbm
    clf = clf.fit(X_train, y_train)

    return clf


def predict_lgbm(data, lgbm_model):
    """
    Inference using LGBM on unknown data
    :param data: Spark dataframe
    :param lgbm_model: LGBM model
    :return: list
    """

    genres_to_keep = ['drama', 'horror', 'biography', 'noir', 'comedy', 'unknown', 'sci-fi']
    cols_to_select = ['genres'] + ['col{}'.format(i) for i in range(19, 34)]

    data = data.select(cols_to_select).withColumn("genres", col("genres").cast("array<string>")) \
        .withColumn("genres_filtered", col("genres") \
                    .array_intersect(genres_to_keep)) \
        .drop("genres") \
        .select(*(col("genres_filtered")[i].alias("genre_{}".format(i)) \
                  for i in range(len(genres_to_keep)))) \
        .fillna(0)
    data = data.drop("col0", "tconst", "primaryTitle", "startYear", "plot")

    data = data.toPandas()

    return lgbm_model.predict(data)


def ensemble(bert_validations, bert_tests, lgbm_validation, lgbm_test):
    """
    Ensemble model using different weights for each model:
    0.125*bert_1 + 0.125*bert_2 + 0.125*bert_3 + 0.125*bert_4 + 0.5*lightGBM
    :param bert_validations: List containing all results for the validation for each BERT model
    :param bert_tests: List containing all results for the hidden test for each BERT model
    :param lgbm_validation: List containing results for the validation test for LGBM model
    :param lgbm_test: List containing results for the hidden test for LGBM model
    :return:
    """
    combined_validation_4berts = np.round(
        np.array(bert_validations[0]) * 0.125 + np.array(bert_validations[1]) * 0.125 + np.array(
            bert_validations[2]) * 0.125 + np.array(bert_validations[3]) * 0.125 + lgbm_validation * 0.5)

    combined_test_4berts = np.round(np.array(bert_tests[0]) * 0.125 + np.array(bert_tests[1]) * 0.125 + np.array(
        bert_tests[2]) * 0.125 + np.array(bert_tests[3]) * 0.125 + lgbm_test * 0.5)

    return combined_validation_4berts, combined_test_4berts
