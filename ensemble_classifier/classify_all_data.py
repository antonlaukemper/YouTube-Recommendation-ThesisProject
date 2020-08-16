from ensemble_classifier.ensemble_classifier import EnsembleClassifier


def save_predictions_CSV(path, data, predictions, confidence_scores, probabilities):
    data = data.drop(columns=['captions',
                              'comments',
                              'snippets'])
    data['predicted_label'] = predictions
    data['probability'] = probabilities
    data['confidence_score'] = confidence_scores
    data.to_csv(path, index=False, float_format='%.3f')

if __name__ == "__main__":
    ensemble = EnsembleClassifier(training_fraction=1)
    ensemble.load_data()
    ensemble.fit(serialized=False)

    captions_unlabeled = ensemble.caption_trainer.get_data(initial=False, training_fraction=1)
    comments_unlabeled = ensemble.comment_trainer.get_data(initial=False, training_fraction=1)
    snippets_unlabeled = ensemble.snippet_trainer.get_data(initial=False, training_fraction=1)
    affiliations_unlabeled = ensemble.affiliation_trainer.get_data(initial=False, training_fraction=1)
    subscriptions_unlabeled = ensemble.subscription_trainer.get_data(initial=False, training_fraction=1)
    cross_comments_unlabeled = ensemble.cross_comment_trainer.get_data(initial=False, training_fraction=1)

    ensemble.add_unlabeled_data(captions_unlabeled,
                                comments_unlabeled,
                                snippets_unlabeled,
                                affiliations_unlabeled,
                                subscriptions_unlabeled,
                                cross_comments_unlabeled)

    unlabeled_probabilities = ensemble.estimate_probabilities(mode='Unlabeled')
    labels, confidence_scores, probabilities = ensemble.predict(unlabeled_probabilities)
    save_predictions_CSV("unlabeled_data_predictions.csv", unlabeled_probabilities, labels, confidence_scores, probabilities)

    test_data = ensemble.estimate_probabilities(mode='Testing')
    labels, confidence_scores, probabilities = ensemble.predict(test_data)
    save_predictions_CSV("test_data_predictions.csv", test_data, labels, confidence_scores, probabilities)
