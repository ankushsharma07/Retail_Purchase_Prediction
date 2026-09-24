def make_prediction(model, data):

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    return prediction, probability