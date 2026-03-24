class TrafficPredictor:
    def __init__(self, model):
        """
        Initializes the TrafficPredictor with a given predictive model.
        
        :param model: A trained predictive model to make predictions.
        """
        self.model = model

    def predict_single(self, data):
        """
        Makes a prediction on a single data point.
        
        :param data: A single input data point in the format expected by the model.
        :return: The predicted value.
        """
        return self.model.predict([data])[0]

    def predict_batch(self, data_list):
        """
        Makes predictions on a batch of data points.
        
        :param data_list: A list of input data points.
        :return: A list of predicted values.
        """
        return self.model.predict(data_list)