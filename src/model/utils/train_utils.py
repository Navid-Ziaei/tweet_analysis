import numpy as np

from src import analyze_across_stanford
from src.model.spicy_model.model import analyze_across_texts_spacy
from src.model.transformer_base_tagging.model_pos_tag import analyze_across_texts_transformers
from src.utils import plot_confusion_matrix, evaluate_model, update_dataframe, save_metrics


def evaluate_ner_models(data, data_handler, labels, paths):
    predicted_labels_trans, ent_type_list_trans = analyze_across_texts_transformers(
        model_name="dslim/bert-base-NER-uncased",
        data=data,
        labels=labels)


    unsupervised_model_names = ['']
    predicted_labels_stan, pos_list_stan, ent_type_list_stan = analyze_across_stanford(data, labels)

    predicted_labels_spacy, pos_list_spacy, ent_type_list_spacy = analyze_across_texts_spacy(
        model_name="en_core_web_trf",
        data=data_handler.data['text'].str.lower().values,
        labels=labels)

    metrics = []
    for model, labels in zip(
            ['Stanford', 'Transformers', 'SpaCy'],
            [predicted_labels_stan, predicted_labels_trans, predicted_labels_spacy]
    ):
        cm, precision, recall, f1_score = evaluate_model(labels, data_handler.data['is_related_to_Across_protocol'])
        plot_confusion_matrix(cm, model, save_path=paths.path_result)
        metrics.append([model, precision, recall, f1_score])
        data_handler.data = update_dataframe(data_handler.data, labels, model)

    predicted_labels = np.array(predicted_labels_spacy) + np.array(predicted_labels_stan) + np.array(
        predicted_labels_trans)
    predicted_labels = (predicted_labels > 0) * 1
    cm, precision, recall, f1_score = evaluate_model(predicted_labels, labels)
    plot_confusion_matrix(cm, 'Combined', save_path=paths.path_result)
    metrics.append(['Combined', precision, recall, f1_score])
    data_handler.data = update_dataframe(data_handler.data, predicted_labels, 'Combined')

    data_handler.data[f'ent_type_SpaCy'] = predicted_labels_spacy
    data_handler.data[f'pos_SpaCy'] = pos_list_spacy
    data_handler.data[f'ent_type_Stanford'] = predicted_labels_stan
    data_handler.data[f'pos_Stanford'] = pos_list_stan
    data_handler.data[f'ent_type_Transformers'] = ent_type_list_trans

    data_handler.data.to_csv(paths.path_result + "updated_data_with_predictions.csv", index=False)
    save_metrics(metrics, paths.path_result + "model_performance_metrics.csv")