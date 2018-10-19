# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

# Note: The following code has been applied from https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own
# and https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/keras_bring_your_own

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import flask
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import load_model

#from environment import create_trainer_environment
prefix = '/opt/ml'
model_path = os.path.join(prefix, 'model')

# Configure the trainer environemnt for SageMaker training
#env = create_trainer_environment()
#print('creating SageMaker trainer environment:\n%s' % str(env))

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = load_model(os.path.join(model_path, 'model.h5'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """
        For the input, do the predictions and return them.

        Args:
            input: The data on which to do the predictions. There will be
                one prediction per row in the dataframe
        """
        sess = K.get_session()
        with sess.graph.as_default():
            clf = cls.get_model()
            return clf.predict(input, batch_size=1)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.
    """
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data.
    """
    data = None

    # Convert from json to numpy
    if flask.request.content_type == 'application/json':
        parsed = json.loads(flask.request.data)
        data = np.array(parsed)
    else:
        return flask.Response(response='This predictor only supports json formatted data', status=415, mimetype='text/plain')

    # Do the prediction
    predictions = ScoringService.predict(data)
    result = json.dumps(predictions.tolist()[0])

    return flask.Response(response=result, status=200, mimetype='application/json')