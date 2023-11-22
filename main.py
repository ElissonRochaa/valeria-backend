import logging

from flask import Flask, request
from flask import jsonify
import datetime
import decimal
import json
from model import Model
from flask_cors import CORS

app = Flask(__name__)

"""
CORS function is to avoid No 'Access-Control-Allow-Origin' error
"""
CORS(app)

@app.route('/')
def hello():
    """webserice test method
    """
    return 'Welcome a Valeria'

@app.route('/predict', methods=['POST'])
def test_get():
    dados = request.get_json()
    model = Model()
    classificacao, probabilidade, explainer = model.predict(dados)
    
    proba = [{'nome':'Dengue', 'probabilidade':probabilidade[1]}, {'nome':'Chikungunya', 'probabilidade':probabilidade[0]}, {'nome':'Inconclusivo', 'probabilidade':probabilidade[2]}]
    response = {'status': 0, 'classificacao': classificacao, 'probabilidades': proba, 'explainer': explainer.to_dict()}
    
    return jsonify(response)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    #app.run(host='192.168.0.58', port=8080, debug=True, processes=4, threaded=True)
    #app.run(threaded=True,debug=True)
    model = Model()
    app.run(host='0.0.0.0', port=5000)
## [END app]
