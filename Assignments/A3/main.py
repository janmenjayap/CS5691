#### Test
import os
import pickle
from statistics import mode
from notebook import MultinomialNaiveBayes


if __name__=='__main__':
    with open('models/feature_extraction.pkl', 'rb') as file:
        feature_extraction = pickle.load(file)

    with open('models/multinomial_nb.pkl', 'rb') as file:
        model_nb_data = pickle.load(file)

    model_nb = MultinomialNaiveBayes(alpha=model_nb_data['alpha'])
    model_nb.class_log_prior_ = model_nb_data['class_log_prior_']
    model_nb.feature_log_prob_ = model_nb_data['feature_log_prob_']
    model_nb.classes_ = model_nb_data['classes_']

    with open('models/svm_linear.pkl', 'rb') as file:
        model_svm_linear = pickle.load(file)

    with open('models/svm_rbf.pkl', 'rb') as file:
        model_svm_rbf = pickle.load(file)

    with open('output.txt', 'w') as output_file:
        output_file.write(f'Predicted Classes:\n1 - SPAM; 0 - HAM.\n\n')
        output_file.write(f'email       predicted\n')
        output_file.write(f'file        class\n')

        # Iterate over each file in the test folder
        for filename in os.listdir('test'):
            if filename.endswith('.txt'):
                # Read email content
                with open(os.path.join('test', filename), 'r') as email_file:
                    email_content = email_file.read()

                # Extract features
                features = feature_extraction.transform([email_content])

                # Apply Naive Bayes model
                predicted_class_nb = model_nb.predict(features)[0]
                predicted_class_svm_linear = model_svm_linear.predict(features)[0]
                predicted_class_svm_rbf = model_svm_rbf.predict(features)[0]

                predicted_class = [predicted_class_nb, predicted_class_svm_linear, predicted_class_svm_rbf]

                # Write output to output.txt
                output_file.write(f'{filename}: {mode(predicted_class)}\n')