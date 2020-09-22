import os
import time
import requests
import time
_DEFAULT_USER_ID = "unknown"


class MLflowTrackingRestApi:
    def __init__(self, hostname, port, experiment_id):
        self.base_url = 'http://' + hostname + ':' + str(port) + '/api/2.0/preview/mlflow'
        self.experiment_id = experiment_id
        self.run_id = self.create_run()

    def create_run(self):
        """Create a new run for tracking."""
        url = self.base_url + '/runs/create'
        # user_id is deprecated and will be removed from the API in a future release
        payload = {'experiment_id': self.experiment_id, 'start_time': int(time.time() * 1000), 'user_id': _get_user_id()}
        r = requests.post(url, json=payload)
        run_id = None
        if r.status_code == 200:
            run_id = r.json()['run']['info']['run_uuid']
        else:
            print("Creating run failed!")
        return run_id

    def list_experiments(self):
        """Get all experiments."""
        url = self.base_url + '/experiments/list'
        r = requests.get(url)
        experiments = None
        if r.status_code == 200:
            experiments = r.json()['experiments']
        return experiments

    def log_param(self, param):
        """Log a parameter dict for the given run."""
        url = self.base_url + '/runs/log-parameter'
        payload = {'run_uuid': self.run_id, 'key': param['key'], 'value': param['value']}
        r = requests.post(url, json=payload)
        return r.status_code

    def log_metric(self, metric):
        """Log a metric dict for the given run."""
        url = self.base_url + '/runs/log-metric'
        payload = {'run_uuid': self.run_id, 'key': metric['key'], 'value': metric['value'], 'step': metric['step']}
        r = requests.post(url, json=payload)
        return r.status_code
     
    def set_tag(self, tag):
        url = self.base_url + '/runs/set-tag'
        payload = {'run_uuid': self.run_id, 'key': tag['key'], 'value': tag['value']}
        r = requests.post(url, json=payload)
        return r.status_code


def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID



# HOSTNAME=""
# PORT=""
# EXPRIMENT_ID=""

# mlflow_rest = MLflowTrackingRestApi(HOSTNAME, PORT, EXPRIMENT_ID)
# param = {'key': 'alpha', 'value': '0.1980'}
# status_code = mlflow_rest.log_param(param)
# metric = {'key': 'precision', 'value': 1}
# status_code = mlflow_rest.log_metric(metric)