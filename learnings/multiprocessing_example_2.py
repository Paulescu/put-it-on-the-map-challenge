from multiprocessing import Pool


class PredictorCoordinates:
    def __init__(self, ):
        pass

    def predict(self):
        pool = Pool()
        output = pool.map(self.worker, range(10))
        return output

    def worker(self, x):
        return self.worker_sub_task(x)

    def worker_sub_task(self, x):
        return x * 2


predictor_coordinates = PredictorCoordinates()
predictor_coordinates.predict()