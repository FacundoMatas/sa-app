import pandas as pd
import pickle
from preprocess import label_feats_from_corpus
from preprocess import split_label_feats
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

# load model, use experiment3
model = pickle.load(open('model.sav', 'rb'))

# app
app = Flask(__name__)
CORS(app)

### swagger specific ###
SWAGGER_URL = ''
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Backend SA APP"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###


# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # get string to predict. Must be a dataframe with 'text' and 'sentiment' columns
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # transform df to an appropriate format for the classifier
    feats = label_feats_from_corpus(data_df)
    example_set = split_label_feats(feats)

    # predictions
    for (name, classifier) in model.items():
        if name == 'Regresion logistica':
            for i, (feats, label) in enumerate(example_set):
                result = classifier.classify(feats)
                for pdist in classifier.prob_classify_many(feats):
                    if result == 0:
                        probability = pdist.prob(0)
                    else:
                        probability = pdist.prob(1)



    # send back to browser
    output = {'results': int(result),
              'probablity': probability}

    # return data
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)