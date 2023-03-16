import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, concat_multiple_documents
from data_manager import transform_raw_data
from training import  save_pipeline, load_pipeline, train_model
from scoring import classification_results
from tqdm import tqdm



def get_parser():
    parser = argparse.ArgumentParser(
        'Document Standartization',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--train', '-t', type=str, default="",
        help="Training of the pipeline of document standartization",
    )
    parser.add_argument(
        '--predict', '-p', type=str, default="",
        help="Runing prediction on the specified path of parsed json document or documents",
    )
    parser.add_argument(
        '--predict_save', '-ps', type=str, default="",
        help="Runing prediction on the specified path of parsed json document and saving csv results",
    )
    parser.add_argument(
        '--summary', '-summ', type=str, default="",
        help="Running prediction on specified path of parsed json documents to print classification report",
    )
    parser.add_argument(
        '--model', '-m', type=str, default="./model.pkl",
        help="Specifying the model path",
    )
    

    return parser


def main():
    parser = get_parser()
    directive = parser.parse_args()

    if directive.train:

        df = transform_raw_data(directive.train)   # read training data
        data = concat_multiple_documents(df)         # concat multiple documents if needed
        data = preprocess(data)                      # apply preprocessing

        X_train, X_test, y_train, y_test = train_test_split(np.array(data["row"]),
                                                            np.array(data["label"]),
                                                            test_size=0.2,
                                                            random_state=42)

        nb = train_model(X_train, y_train)
        save_pipeline(pipeline=nb, save_path="./nb_model.pkl")

    elif directive.predict_save:

        model = load_pipeline(file_path="./model.pkl")

        test_data = transform_raw_data(directive.predict_save)
        index = 1
        final_results = pd.DataFrame(columns=["document", "header_rows", "header_count"])

        for data in test_data:
            data = preprocess(data)
            y_pred = model.predict(data['row'])

            data['predicted_header'] = y_pred
            headers_list = list(data[data['predicted_header'] == 1].index)
            final_results = final_results.append(
                {"document": index, "header_rows": headers_list, "header_count": len(headers_list)}, ignore_index=True)
            index += 1

        final_results.to_csv("./headers_prediction_results.csv", index=False)
        
    elif directive.predict:

        trained_model = load_pipeline(file_path=directive.model)
        
        with open(directive.predict, encoding="utf8") as f:
            lines = [eval(line.rstrip()) for line in f]

        for document in tqdm(lines):

            values = []
            index = 0

            for line_dict in document:

                values.append([])

                for value in line_dict['values']:
                    values[index].append(value['value'])

                current_row = " ".join((" ".join(values[index])).split())

                prediction = (trained_model.predict([current_row]))[0]
                if prediction == 1:             
                    line_dict['type'] = "HEADERS"
                else:
                    line_dict['type'] = "OTHER"

                index += 1

            with open("./test_results.txt", "w+") as f:
                f.write('\n'.join(str(line) for line in lines))

    elif directive.summary:

        model = load_pipeline(file_path=directive.model)
        
        try:
            documents = transform_raw_data(directive.summary)
            test_data = concat_multiple_documents(documents)
            test_data = preprocess(test_data)
            classification_results(saved_model=model, y_test=test_data)

        except Exception as e:
            print(e)
    else:
        print("You have specified wrong argument")


if __name__ == '__main__':
    main()
